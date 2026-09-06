---
source_kind: wiki
source_name: confluence
source_id: "1002"
title: "Platform service level objectives"
url: https://example.invalid/wiki/1002
author: Marcus Hale
author_side: customer
created_at: 2025-09-01T10:00:00+00:00
updated_at: 2026-01-15T14:10:00+00:00
corpus_role: context
item_label: Page
---

These objectives apply to all authenticated pages of the product.

## Availability

The platform targets 99.9% monthly availability for authenticated traffic. This
target covers every page reachable after sign-in, including administrative and
reporting pages.

## Latency

A page of list data returns its first byte within 400 ms at the 95th percentile
under the normal working-hours load of 2,000 concurrent sessions.

## Reporting and log views

Reporting and log views are held to the same availability target as the rest of
the authenticated product. They may use pagination to meet the latency target.
