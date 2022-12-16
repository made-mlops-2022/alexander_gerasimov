from pydantic import BaseModel


class HeartResponse(BaseModel):
    id: int
    condition: int
