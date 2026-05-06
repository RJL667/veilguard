"""SharePoint connector — Veilguard's first real connector.

Implements the Connector contract against Microsoft Graph + SharePoint:

  * ``hint(prompt)``      — Graph /search/query (driveItem entityType)
  * ``search(query)``     — same as hint, returns refs only
  * ``read(ref)``         — Graph /drives/{drive}/items/{id}/content +
                            content extraction (txt/md/docx/pdf)
  * ``list(path)``        — Graph drive item listing
  * ``get_permissions()`` — Graph /drives/{drive}/items/{id}/permissions

OAuth tokens are resolved per-user via the connector framework's
:class:`CredentialResolver` — production deployments wire the
:class:`HttpCredentialResolver` against LibreChat's internal token
endpoint.
"""
from .connector import SharePointConnector

__all__ = ["SharePointConnector"]
