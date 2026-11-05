# Incident runbook

1. Acknowledge the alert in the on-call channel within 5 minutes.
2. Check the status page and the last deploy in `docs/release-process.md`.
3. If the API error rate is above 2%, roll back to the previous tag.
4. Open an incident note in `notes/` with a timeline.
5. After recovery, schedule a blameless review within 3 working days.

Escalation: platform@acme.example, then the engineering manager.
