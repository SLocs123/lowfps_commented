from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated


class Reid_config(BaseModel):
    """Settings for the ReID model used to compare object appearance."""

    reid_config_path: str = 'ByteTrack/feature_extractor/sbs_R50-ibn.yml'
    reid_weight: str = 'ByteTrack/feature_extractor/veri_sbs_R50-ibn.pth'
    reid_device: str = 'cuda'


class RedisConfig(BaseModel):
    """Unused here, but kept from the original project configuration model."""

    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str
    input_stream_prefix: str = 'objectdetector'
    output_stream_prefix: str = 'featureextractor'


class TrackletDataBase(BaseModel):
    """Settings placeholder for track storage outside the current script."""

    searching_time: int
    lost_time: int


class FeatureExtrator(BaseSettings):
    """Top-level settings object passed into the `Extractor` class."""

    frame_info: bool = False
    reid_config: Reid_config
    max_queue_size: int = 64
    sampling_rate: int = 5
    last_frame_id: int = 8999

    # Allow nested environment variables such as `REID_CONFIG__REID_DEVICE=cpu`.
    model_config = SettingsConfigDict(env_nested_delimiter='__')