---
source_kind: tracker
source_name: jira
source_id: SCOPE-1
title: "Workspace sharing and audit log"
url: https://example.invalid/browse/SCOPE-1
author: Dana Reyes
author_side: customer
created_at: 2026-03-02T09:14:00+00:00
updated_at: 2026-04-11T16:20:00+00:00
corpus_role: requirement
item_label: Epic
status: In Progress
---

Workspaces today are private to the person who created them. Customers on the
Team plan have asked to share a workspace with colleagues and to see who changed
what afterwards.

## Goal

Let a workspace owner invite colleagues, give them a level of access, and review
a log of actions taken inside the workspace.

## Stories

Sharing splits into invitations, access levels, the audit log itself, export of
the log, and migration of the sharing rows that already exist in the legacy
`shared_links` table.

## Notes

The Team plan already exists and is billed through the current subscription
flow; nothing about billing changes in this epic.
