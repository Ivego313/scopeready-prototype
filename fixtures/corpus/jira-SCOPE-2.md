---
source_kind: tracker
source_name: jira
source_id: SCOPE-2
title: "Access levels for shared workspaces"
url: https://example.invalid/browse/SCOPE-2
author: Dana Reyes
author_side: customer
created_at: 2026-03-04T11:02:00+00:00
corpus_role: requirement
item_label: Story
parent_source_id: SCOPE-1
status: In Progress
---

A workspace has exactly three access levels.

## Levels

- **Owner** — the person who created the workspace. May invite and remove
  members, change anyone's level, rename the workspace and delete it. There is
  always exactly one owner.
- **Editor** — may create, edit and delete documents inside the workspace, and
  may see the member list. May not invite, remove members or delete the
  workspace.
- **Viewer** — may open and read documents and may export them. May not change
  any document and may not see the audit log.

Ownership can be transferred by the current owner to any editor. Removing a
member revokes their access immediately, including any open session.

## Acceptance criteria

- An editor who opens the workspace settings page sees the member list but no
  invite control.
- A viewer who requests a document edit endpoint directly receives 403.
- After ownership transfer the previous owner holds editor rights.
