from typing import List, Optional
from pydantic import BaseModel

class NodeData(BaseModel):
    name: Optional[str] = None
    level: Optional[int] = None
    sex: Optional[str] = None
    display_name: Optional[str] = None
    diseases: Optional[List[str]] = None
    coordinates: Optional[List[float]] = None
    center: Optional[List[float]] = None
    mother: Optional[str] = None
    father: Optional[str] = None
    partners: Optional[List[str]] = None
    divorced: Optional[List[str]] = None
    noparents: Optional[bool] = None
    top_level: Optional[bool] = None
    status: Optional[int] = None
    dob: Optional[str] = None
    age: Optional[str] = None
    adopted_in: Optional[bool] = None
    adopted_out: Optional[bool] = None
    mztwin: Optional[int] = None
    dztwin: Optional[int] = None
    carrier: Optional[bool] = None
    proband: Optional[bool] = None
    shading: Optional[List[str]] = None
    additional_info: Optional[List[str]] = None
    miscarriage: Optional[bool] = None
    stillbirth: Optional[bool] = None
    termination: Optional[bool] = None