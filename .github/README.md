# Repository Automation

Role: issue templates and mirror-CI configuration.

## Current state

- Issue forms live in `ISSUE_TEMPLATE/`.
- The GitHub Actions workflow lives in `workflows/`; the primary repository may run on GitLab.

## TODO

- [ ] Decide whether GitHub Actions remains mirror CI or checks move to `.gitlab-ci.yml`.
- [ ] Keep issue forms and CI commands aligned with the supported Freeciv/MuZero workflow.
- [ ] Document required CI secrets, runners, caches, and artifact retention.
