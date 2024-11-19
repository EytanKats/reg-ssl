from domain_adversarial_segmentation.core.post_processor_vis import PostProcessorVis
from domain_adversarial_segmentation.core.post_processor_masks import PostProcessorMasks
from domain_adversarial_segmentation.core.post_processor_distributions import PostProcessorDistributions
from domain_adversarial_segmentation.core.post_processor_masks_full_body import PostProcessorMasksFullBody


def get_post_processor(settings):

    if settings['postprocessor']['mode'] == 'distributions':
        post_processor = PostProcessorDistributions
    elif settings['postprocessor']['mode'] == 'masks':
        post_processor = PostProcessorMasks
    elif settings['postprocessor']['mode'] == 'masks_full_body':
        post_processor = PostProcessorMasksFullBody
    elif settings['postprocessor']['mode'] == 'vis':
        post_processor = PostProcessorVis

    return post_processor
