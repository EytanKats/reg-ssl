import os
import torch

from simple_converge.apps import SingleModelApp

from core.data_utils import augment_affine_nl, normalize_img
from core.registration_pipeline import update_fields


class SingleModelAppEx(SingleModelApp):

    def __init__(
            self,
            settings,
            mlops_task,
            architecture,
            loss_function,
            metric,
            scheduler,
            optimizer,
            sampling_dataloader
    ):

        super(SingleModelAppEx, self).__init__(
            settings,
            mlops_task,
            architecture,
            loss_function,
            metric,
            scheduler,
            optimizer
        )

        # architecture function is used to reinitialize the model after generation of pseudo-labels for the first time
        self.architecture = architecture

        # sampling dataloader contains train data
        # we iterate over it at the beginning of each epoch to generate pseudo-labels
        self.sampling_dataloader = sampling_dataloader

        # pseudo labels (pseudo displacement fields regenerated on the start of the epoch)
        self.pseudo_labels = None

        # volume dimensions
        self.d1 = self.settings['dataset']['d1']
        self.d2 = self.settings['dataset']['d2']
        self.d3 = self.settings['dataset']['d3']

        # placeholders for batch of images
        self.img0 = torch.zeros(self.settings['dataloader']['batch_size'], 1, self.d1, self.d2, self.d3).to(self.device)
        self.img1 = torch.zeros(self.settings['dataloader']['batch_size'], 1, self.d1, self.d2, self.d3).to(self.device)
        self.img0_aug = torch.zeros(self.settings['dataloader']['batch_size'], 1, self.d1, self.d2, self.d3).to(self.device)
        self.img1_aug = torch.zeros(self.settings['dataloader']['batch_size'], 1, self.d1, self.d2, self.d3).to(self.device)

        # placeholders for batch of pseudo labels
        self.target = torch.zeros(self.settings['dataloader']['batch_size'], 3, self.d1 // 2, self.d2 // 2, self.d3 // 2).to(self.device)
        self.target_aug = torch.zeros(self.settings['dataloader']['batch_size'], 3, self.d1 // 2, self.d2 // 2, self.d3 // 2).to(self.device)

        # placeholders for batch of affine transformations
        self.affine0 = torch.zeros(self.settings['dataloader']['batch_size'], self.d1, self.d2, self.d3, 3).to(self.device)
        self.affine1 = torch.zeros(self.settings['dataloader']['batch_size'], self.d1, self.d2, self.d3, 3).to(self.device)

    def on_epoch_start(self, epoch):

        # Generate pseudo-labels
        self.pseudo_labels, _, _, d_all_adam, d_all_ident, sdlogj, sdlogj_adam = update_fields(
            self.sampling_dataloader,
            self.model,
            use_adam=True,
            num_warps=self.settings['app']['num_warps'],
            ice=self.settings['app']['ice'],
            reg_fac=self.settings['app']['reg_factor'],
            compute_jacobian=self.settings['app']['compute_jacobian'],
            num_labels=self.settings['app']['num_labels'],
            clamp=self.settings['app']['clamp']
        )

        # Reinitialize model after generation of pseudo-labels for the first time
        if epoch == 0:
            self.model = self.architecture(self.settings)

    def on_epoch_end(self, epoch, is_plateau=False):
        super(SingleModelAppEx, self).on_epoch_end(epoch, is_plateau)

    def training_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        self.model.train()

        # get images and corresponding pseudo-labels
        indices = self.parse_batch_data(data)
        for j in range(self.settings['dataloader']['batch_size']):
            self.img0[j:j + 1] = normalize_img(self.img0[j:j + 1])
            self.img1[j:j + 1] = normalize_img(self.img1[j:j + 1])

        for j in range(self.settings['dataloader']['batch_size']):
            self.target[j:j + 1] = self.pseudo_labels[indices[j]:indices[j] + 1].to(self.device)

        # augment
        for j in range(self.settings['dataloader']['batch_size']):
            self.target_aug[j:j + 1], self.affine0[j:j + 1], self.affine1[j:j + 1] = augment_affine_nl(self.target[j:j + 1], shape=(1, 1, self.d1, self.d2, self.d3))
            self.img0_aug[j:j + 1] = torch.nn.functional.grid_sample(self.img0[j:j + 1], self.affine0[j:j + 1])
            self.img1_aug[j:j + 1] = torch.nn.functional.grid_sample(self.img1[j:j + 1], self.affine1[j:j + 1])

        # Get pseudo labels for MRI data, use teacher
        if self.settings['app']['use_mean_teacher']:
            self.ema.apply_shadow()
            self.model.eval()

            with torch.no_grad():
                mri_pseudo_labels, _ = self.model(mri_data)
                mri_pseudo_labels = torch.softmax(mri_pseudo_labels / self.settings['app']['temperature'], dim=1)

        if self.settings['app']['use_mean_teacher']:
            self.ema.restore()
            self.model.train()

        self.optimizer.zero_grad()
        self.cls_optimizer.zero_grad()

        # Apply model on CT data
        if self.settings['app']['use_gin']:
            os.environ["DG_TTA_INTERNAL_AUGMENTATION"] = "true"
        ct_model_output, ct_feature_map = self.model(ct_data)
        if self.settings['app']['use_gin']:
            os.environ["DG_TTA_INTERNAL_AUGMENTATION"] = "false"

        # Apply student model on MRI data
        if self.settings['app']['use_mean_teacher']:

            if self.settings['app']['apply_affine_augmentation']:

                identity_grid = torch.nn.functional.affine_grid(
                    torch.eye(4, device='cuda').repeat(self.settings['dataloader']['batch_size'], 1, 1)[:, :3],
                    [self.settings['dataloader']['batch_size'], 1] + list(self.settings['model']['img_size']),
                    align_corners=False,
                )
                zero_grid = 0.0 * identity_grid
                grid = zero_grid
                grid_inverse = zero_grid

                R, R_inverse = get_rand_affine(self.settings['dataloader']['batch_size'], flip=False)
                R, R_inverse = R.to('cuda'), R_inverse.to('cuda')

                grid = grid + (
                        torch.nn.functional.affine_grid(R, [self.settings['dataloader']['batch_size'], 1] + list(self.settings['model']['img_size']), align_corners=False)
                        - identity_grid
                )
                grid_inverse = grid_inverse + (
                        torch.nn.functional.affine_grid(R_inverse, [self.settings['dataloader']['batch_size'], 1] + list(self.settings['model']['img_size']), align_corners=False)
                        - identity_grid
                )

                grid = grid + identity_grid
                mri_data_aug = torch.nn.functional.grid_sample(
                    mri_data, grid, padding_mode="border", align_corners=False
                )

                mri_model_output_aug, mri_feature_map = self.model(mri_data_aug)

                grid_inverse = grid_inverse + identity_grid
                mri_model_output = torch.nn.functional.grid_sample(
                    mri_model_output_aug, grid_inverse, align_corners=False
                )

            else:
                mri_model_output = self.model(mri_data)

        if self.settings['app']['enable_classifier_loss']:

            if not self.settings['app']['use_mean_teacher']:  # prevent double call of the model
                _, mri_feature_map = self.model(mri_data)

            if self.settings['app']['reverse_gradients']:
                mri_feature_map = GradientReversalLayer.apply(mri_feature_map, alpha)
                ct_feature_map = GradientReversalLayer.apply(ct_feature_map, alpha)

            mri_domain_classification = self.domain_classifier(mri_feature_map)
            ct_domain_classification = self.domain_classifier(ct_feature_map)

            if self.settings['model']['classifier_type'] == 'image':
                ct_domain_labels = torch.zeros(ct_domain_classification.shape[0]).long().cuda()
                mri_domain_labels = torch.ones(mri_domain_classification.shape[0]).long().cuda()

            elif self.settings['model']['classifier_type'] == 'pixel':
                dim = self.settings['model']['pixel_classifier_output_size']
                batch_size = self.settings['dataloader']['batch_size']
                ct_domain_labels = torch.zeros(batch_size, dim, dim, dim).long().cuda()
                mri_domain_labels = torch.ones(batch_size, dim, dim, dim).long().cuda()

        # Calculate loss
        batch_loss_list = list()

        ct_labels[ct_labels == -1] = 0  # todo: check validity of data preprocessed for nnunet
        dice_loss = self.losses_fns[0](ct_model_output, ct_labels)
        batch_loss_list.append(dice_loss.detach().cpu().numpy())
        loss = dice_loss

        if self.settings['app']['enable_classifier_loss']:
            ce_loss = self.losses_fns[1](ct_domain_classification, ct_domain_labels) + self.losses_fns[1](mri_domain_classification, mri_domain_labels)
            batch_loss_list.append(ce_loss.detach().cpu().numpy())
            loss += self.settings['app']['classifier_loss_weight'] * ce_loss
        else:
            batch_loss_list.append(0)

        if self.settings['app']['use_mean_teacher']:
            pseudo_dice_loss = self.losses_fns[2](mri_model_output, mri_pseudo_labels)
            batch_loss_list.append(pseudo_dice_loss.detach().cpu().numpy())
            loss += pseudo_dice_loss
        else:
            batch_loss_list.append(0)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        if self.settings['app']['enable_classifier_loss']:
            self.cls_optimizer.step()

        with torch.no_grad():
            # Calculate metrics
            batch_metric_list = list()

            for idx in range(len(self.settings['dataset']['labels']) + 1):
                metric = self.metrics_fns[idx](ct_model_output.detach(), ct_labels.detach())
                batch_metric_list.append(metric.detach().cpu().numpy())

            if self.settings['app']['enable_classifier_loss']:
                domain_classification = torch.cat((ct_domain_classification, mri_domain_classification), 0)
                domain_labels = torch.cat((ct_domain_labels, mri_domain_labels), 0)
                metric = self.metrics_fns[-1](domain_classification, domain_labels)
                batch_metric_list.append(metric.detach().cpu().numpy())
            else:
                batch_metric_list.append(0)

        if self.settings['app']['use_ema']:
            self.ema.update()
            self.cls_ema.update()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        self.model.eval()
        self.domain_classifier.eval()

        if self.settings['app']['use_ema']:
            self.ema.apply_shadow()
            self.cls_ema.apply_shadow()

        ct_data, ct_labels, mri_data = self.parse_batch_data(data)
        ct_data = ct_data.to(self.device)
        ct_labels = ct_labels.to(self.device)
        mri_data = mri_data.to(self.device)

        with torch.no_grad():

            # Apply model
            ct_model_output, ct_feature_map = self.model(ct_data)

            if self.settings['app']['enable_classifier_loss']:
                _, mri_feature_map = self.model(mri_data)
                ct_domain_classification = self.domain_classifier(ct_feature_map)
                mri_domain_classification = self.domain_classifier(mri_feature_map)

                if self.settings['model']['classifier_type'] == 'image':
                    ct_domain_labels = torch.zeros(ct_domain_classification.shape[0]).long().cuda()
                    mri_domain_labels = torch.ones(mri_domain_classification.shape[0]).long().cuda()
                elif self.settings['model']['classifier_type'] == 'pixel':
                    dim = self.settings['model']['pixel_classifier_output_size']
                    batch_size = 1
                    ct_domain_labels = torch.zeros(batch_size, dim, dim, dim).long().cuda()
                    mri_domain_labels = torch.ones(batch_size, dim, dim, dim).long().cuda()

            # Calculate loss
            batch_loss_list = list()

            ct_labels[ct_labels == -1] = 0  # todo: check validity of data preprocessed for nnunet
            dice_loss = self.losses_fns[0](ct_model_output, ct_labels)
            batch_loss_list.append(dice_loss.detach().cpu().numpy())

            if self.settings['app']['enable_classifier_loss']:
                ce_loss = self.losses_fns[1](ct_domain_classification, ct_domain_labels) + self.losses_fns[1](mri_domain_classification, mri_domain_labels)
                batch_loss_list.append(ce_loss.detach().cpu().numpy())
            else:
                batch_loss_list.append(0)

            batch_loss_list.append(0)  # placeholder for pseudo_dice mri loss

            # Calculate metrics
            batch_metric_list = list()

            for idx in range(len(self.settings['dataset']['labels']) + 1):
                metric = self.metrics_fns[idx](ct_model_output, ct_labels)
                batch_metric_list.append(metric.detach().cpu().numpy())

            if self.settings['app']['enable_classifier_loss']:
                domain_classification = torch.cat((ct_domain_classification, mri_domain_classification), 0)
                domain_labels = torch.cat((ct_domain_labels, mri_domain_labels), 0)
                metric = self.metrics_fns[-1](domain_classification, domain_labels)
                batch_metric_list.append(metric.detach().cpu().numpy())
            else:
                batch_metric_list.append(0)

        if self.settings['app']['use_ema']:
            self.ema.restore()
            self.cls_ema.restore()

        return batch_loss_list, batch_metric_list

    def predict(self, data):

        self.model.eval()
        self.domain_classifier.eval()
        image = data['mri']['image'].to(self.device)

        if self.settings['app']['load_nnunet_ckpt']:
            self.predictor.network = self.model
            if self.settings['test']['checkpoints'][0]:
                self.predictor.list_of_parameters = [torch.load(self.settings['test']['checkpoints'][0])['model']]
            output = self.predictor.predict_logits_from_preprocessed_data(image[0]).unsqueeze(0)
        else:
            with torch.no_grad():
                if self.settings['postprocessor']['mode'] == 'distributions':
                    _, feature_map = self.model(image)
                    output = self.domain_classifier(feature_map)
                elif self.settings['postprocessor']['mode'] == 'masks' or self.settings['postprocessor']['mode'] == 'masks_full_body' or self.settings['postprocessor']['mode'] == 'vis':
                    output, _ = self.val_model(image)

        return output.detach().cpu().numpy()

    def parse_batch_data(self, data):

        indices = data['idx'].numpy().tolist()

        for j in range(self.settings['dataloader']['batch_size']):
            self.img0[j:j + 1] = data['image_1'][j:j + 1].to(self.device)
            self.img1[j:j + 1] = data['image_2'][j:j + 1].to(self.device)

        return indices



