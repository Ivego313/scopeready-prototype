---
source_kind: wiki
source_name: confluence
source_id: "1001"
title: "Data retention and deletion policy"
url: https://example.invalid/wiki/1001
author: Dana Reyes
author_side: customer
created_at: 2025-11-20T12:00:00+00:00
updated_at: 2026-02-02T09:30:00+00:00
corpus_role: context
item_label: Page
---

This policy applies to every product surface and every feature built on the
platform. Teams do not restate it in individual tickets.

## Deletion on request

When an account holder asks for deletion, the account record, its documents and
its workspace memberships are removed within 30 days of the request. Audit log
entries naming the deleted person are rewritten to a tombstone identifier rather
than deleted, because an audit log with holes in it is not an audit log.

## Backups and derived copies

Encrypted backups are retained for 35 days and are not selectively edited. A
deletion request is therefore complete once the last backup covering the record
has expired. Analytics events are pseudonymised at collection and are not
re-linked to a deleted account.

## Export

An account holder may request a machine-readable export of their own data,
delivered as a ZIP of JSON files within 14 days.

## Retention of operational data

Application logs are kept for 90 days. Audit logs are kept for the life of the
workspace plus one year.
