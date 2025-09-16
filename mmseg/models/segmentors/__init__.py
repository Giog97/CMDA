from .base import BaseSegmentor, BaseSegmentorEvents, BaseSegmentorFusion
from .encoder_decoder import EncoderDecoder, EventsEncoderDecoder, FusionEncoderDecoder
from .encoder_decoder_originale import EncoderDecoder_originale, EventsEncoderDecoder_originale, FusionEncoderDecoder_originale # aggiunto '_originale'

__all__ = ['BaseSegmentor', 'BaseSegmentorEvents', 'BaseSegmentorFusion',
           'EncoderDecoder', 'EventsEncoderDecoder', 'FusionEncoderDecoder',
           'EncoderDecoder_originale', 'EventsEncoderDecoder_originale','FusionEncoderDecoder_originale'] # aggiunto '_originale'
