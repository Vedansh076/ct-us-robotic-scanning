# Project-Scoped Rules: CT-to-Ultrasound Robotic Scanning

This workspace contains two tracking files to preserve context and tasks across sessions:
1. [agent.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/agent.md) tracks the overall project state, findings, and context.
2. [task.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/task.md) tracks the checklist of pending and completed tasks.

## Rules for Assistant Agents:
1. **Context Check on Startup:** On your very first turn of any conversation or task in this workspace, you MUST read both `agent.md` and `task.md` using `view_file` to establish context.
2. **Synchronize Tasks:** When you start or complete any task listed in `task.md`, update its checklist status (e.g. `[x]`, `[/]`, `[ ]`) immediately.
3. **Log Technical Findings:** If you discover new bugs, architectural details, or normalisation/activation mismatches, document them in Section 2 of `agent.md`.
4. **Active Workspace:** Make all changes in the `ct_us` main project folder. Do not work in `ct_us_standalone` directly unless explicitly requested to update the standalone package.
