---
source_kind: tracker
source_name: jira
source_id: SCOPE-31
title: "Transactional email provider limits"
url: https://example.invalid/browse/SCOPE-31
author: Priya Nandakumar
author_side: executor
created_at: 2026-02-18T09:00:00+00:00
corpus_role: requirement
item_label: Story
parent_source_id: SCOPE-30
status: Done
---

All transactional email goes through Postmark. The account is owned by the
customer and the API token is supplied by the customer; the executor never holds
production credentials.

The plan allows 300 messages per minute and 100,000 per month. Above the per
minute limit Postmark returns 429 and the sender must retry with backoff.

When Postmark is unavailable, outbound messages are queued for up to one hour
and then reported as failed to the requesting user.
