# Release process

Releases go out every second Thursday.

1. Freeze `main` at 10:00 and tag a release candidate.
2. Run the full test suite and the staging smoke test.
3. Write release notes summarising user-facing changes.
4. Deploy to production after sign-off from the release owner.

Rollbacks: redeploy the previous tag; database migrations must be backwards compatible.
