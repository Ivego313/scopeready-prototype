---
source_kind: tracker
source_name: jira
source_id: SCOPE-4
title: "Audit log and export"
url: https://example.invalid/browse/SCOPE-4
author: Priya Nandakumar
author_side: executor
created_at: 2026-03-09T13:25:00+00:00
corpus_role: requirement
item_label: Story
parent_source_id: SCOPE-1
status: To Do
---

Every action that changes a workspace or its membership is written to an audit
log: document created, edited, deleted, member invited, member removed, level
changed, ownership transferred.

An entry records who acted, what they acted on, and when, in UTC.

## Acceptance criteria

- The owner can filter the log by member and by date range.
- The owner can export the filtered log as CSV.
- An export of an empty result set produces a file with only the header row.
- Deleting a document does not delete its audit entries.
