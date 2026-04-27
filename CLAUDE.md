# CourtVision — Claude Code Session

On every session start, read these files before doing anything else:
- CONTEXT.md
- TASKS.md

## When writing new code
- Follow the coding style guide at ./docs/coding-style-guide.md

## Your role
- Architecture review and ADR decisions
- Code review: MVVM compliance, coroutine scoping, thread safety
- Documentation and planning

## Documentation and planning
- Plan files (`docs/plans/`) must not reference `TASKS.md` — it is temporary task tracking and will be archived. Any constraint, risk, or context worth recording belongs in the plan doc itself.

## What to leave alone
- Do not introduce new dependencies without flagging it
- Do not redesign architecture mid-session without explicit discussion

## When touching ML pipeline files
Also read ./docs/tdd.md before proceeding.

## When commiting changes or pushing to remote
- Review ./docs/dev-workflow.md before proceeding. Instruct user to validate.

## Build & Test

**Do NOT run Gradle, build commands, or tests yourself.** Your sandbox environment lacks the correct JDK, writable Gradle home, and Android SDK paths — builds will fail with environment errors, not code errors.

Instead:
- Write the code and tests
- Tell the user exactly which test commands to run (class names, flags)
- The user will run them locally and paste back results
- Do NOT create workaround directories (.gradle-user, .gradle-user-home, etc.) in the repo
