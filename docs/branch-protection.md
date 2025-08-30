# Branch protection

The `main` branch requires pull requests and enforces a linear history.

## Required status checks
- tests (py3.11)
- tests (py3.12)
- security / bandit
- security / semgrep
- Semantic PR Title

These checks must be green and the branch must be up-to-date before merge.

## Merge policy
- Squash merge is enabled; merge commits are disallowed.
- Auto-merge may be enabled when all checks pass and an approving review is present.
- Head branches are deleted automatically after merge.

## Adding a new required check
1. Add the check to the CI workflow.
2. Merge it once to `main` without enforcing it.
3. Update branch protection to include the new check name.
4. Remove it when no longer needed.
