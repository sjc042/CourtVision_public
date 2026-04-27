On every session start, read these files before doing anything else:
@./CONTEXT.md
@./TASKS.md

## When writing new code
- Follow the coding style guide at ./docs/coding-style-guide.md

## Your role
- Jetpack Compose UI and screen layouts
- Logcat debugging (paste output directly)
- Screenshot-based UI debugging
- Performance analysis from benchmark output

## Rules
- Do not touch pipeline or ML layer files
- Flag any suggested dependency additions
- Plan files (`docs/plans/`) must not reference `TASKS.md` — it is temporary task tracking and will be archived. Any constraint, risk, or context worth recording belongs in the plan doc itself.

## When performing code review
- Review ./docs/dev-workflow.md first.