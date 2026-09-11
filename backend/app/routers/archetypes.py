"""Production V3.2C player-style archetype catalogue endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.archetypes import archetype_catalogue_response
from app.dependencies import get_db
from app.schemas import ArchetypeCatalogueResponse

router = APIRouter(prefix="/api/archetypes", tags=["archetypes"])


@router.get("", response_model=ArchetypeCatalogueResponse)
def get_archetypes(
    session: Annotated[Session, Depends(get_db)],
) -> ArchetypeCatalogueResponse:
    return archetype_catalogue_response(session)
