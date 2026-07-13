# Project-Scoped Rules: CT-to-Ultrasound Robotic Scanning

This workspace contains three tracking files to preserve context, tasks, and execution commands across sessions:
1. [agent.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/agent.md) tracks the overall project state, findings, and context.
2. [task.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/task.md) tracks the checklist of pending and completed tasks.
3. [commands.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/commands.md) tracks all execution, training, and evaluation commands across all models.

## Rules for Assistant Agents:
1. **Context Check on Startup:** On your very first turn of any conversation or task in this workspace, you MUST read `agent.md`, `task.md`, and `commands.md` using `view_file` to establish context.
2. **Synchronize Tasks:** When you start or complete any task listed in `task.md`, update its checklist status (e.g. `[x]`, `[/]`, `[ ]`) immediately.
3. **Log Technical Findings:** If you discover new bugs, architectural details, or normalisation/activation mismatches, document them in Section 2 of `agent.md`.
4. **Maintain Command Reference:** Whenever adding or modifying executable scripts, model architectures, CLI flags, or execution workflows, you MUST update and maintain [commands.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/commands.md).
5. **Active Workspace:** Make all changes in the `ct_us` main project folder. Do not work in `ct_us_standalone` directly unless explicitly requested to update the standalone package.
6. **Git Synchronization:** Upon completing a task or stage, verify compilation. Stage the modified code files, commit them, and push the updates to GitHub using `git push origin main` to keep the remote repository current.
