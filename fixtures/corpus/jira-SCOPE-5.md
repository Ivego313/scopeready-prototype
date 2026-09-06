---
source_kind: tracker
source_name: jira
source_id: SCOPE-5
title: "Migrate legacy shared_links rows"
url: https://example.invalid/browse/SCOPE-5
author: Priya Nandakumar
author_side: executor
created_at: 2026-03-12T10:05:00+00:00
corpus_role: requirement
item_label: Story
parent_source_id: SCOPE-1
status: To Do
---

The current product shares a workspace through a row in the legacy
`shared_links` table, written by the 2021 sharing feature. Roughly 40,000 rows
exist. Each row holds a workspace id, a target email and a boolean `can_edit`.

These rows must become memberships: `can_edit = true` becomes an editor,
`can_edit = false` becomes a viewer.

The legacy table is written by a PHP service that is still in production and
cannot be changed as part of this epic.
