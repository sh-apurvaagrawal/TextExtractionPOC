from pydantic import BaseModel, HttpUrl

class ImageRequest(BaseModel):
    image_id: str
    s3_url: HttpUrl
    apply_orientation_correction: bool = True
