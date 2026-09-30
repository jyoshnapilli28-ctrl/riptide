"""
Unit tests for Dynamic Structure-Aware Email Chunker.
Ensures zero fixed-size chunking and verifies metadata generation.
"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.chunker import chunk_email, DynamicEmailChunker


SAMPLE_EMAIL_1 = r"""Message-ID: <123456.1075840001.JavaMail.evans@thyme>
Date: Wed, 12 Dec 2001 09:14:00 -0800 (PST)
From: kenneth.lay@enron.com
To: jeff.skilling@enron.com, greg.whalley@enron.com
Subject: Q4 Risk Committee Review & Pipeline Strategy
X-Folder: \\KLAY (Non-Privileged)\\Lay, Kenneth\\Sent Items

Jeff, Greg,

Please find attached the updated briefing notes for tomorrow morning's risk committee session. We need to evaluate the mark-to-market balances and ensure liquidity reserves remain aligned with board covenants.

The Western power markets have experienced substantial volatility this week. Our trading desks have adjusted basis hedges accordingly, but counterparty exposure limits must be enforced without exception.

Let me know if you would like to convene 15 minutes prior to review the executive summary.

Best regards,
Ken Lay
Chairman and CEO
Enron Corp.
"""

SAMPLE_EMAIL_WITH_FORWARD = """Message-ID: <987654.1075840002.JavaMail.evans@thyme>
Date: Thu, 13 Dec 2001 14:22:00 -0800 (PST)
From: mark.haedicke@enron.com
To: legal.team@enron.com
Subject: Fwd: FERC Filing Deadlines
X-From: Haedicke, Mark

Team, please review the forwarded notification from regulatory affairs.

-----Original Message-----
From: richard.sanders@enron.com
Sent: Thursday, December 13, 2001 1:45 PM
To: mark.haedicke@enron.com
Subject: FERC Filing Deadlines

Mark,

The FERC commissioners have requested supplemental testimony regarding the California energy procurement protocols. We need all external filings submitted by 5:00 PM EST next Monday.

Regards,
Richard B. Sanders
Vice President, Legal
"""


def test_chunk_email_structure():
    chunks = chunk_email(SAMPLE_EMAIL_1, email_id="enron_00001")
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    # Check header chunk
    header_chunk = chunks[0]
    assert header_chunk["chunk_type"] == "header_summary"
    assert "Subject: Q4 Risk Committee Review" in header_chunk["text"]
    assert "From: kenneth.lay@enron.com" in header_chunk["text"]
    assert header_chunk["chunk_reason"] == "rfc822_header_envelope"

    # Check body paragraph chunks
    body_chunks = [c for c in chunks if c["chunk_type"] == "body_paragraph"]
    assert len(body_chunks) >= 1
    assert any("Western power markets" in c["text"] for c in body_chunks)

    # Check signature chunk
    sig_chunks = [c for c in chunks if c["chunk_type"] == "signature_block"]
    assert len(sig_chunks) == 1
    assert "Ken Lay" in sig_chunks[0]["text"]
    assert "Chairman and CEO" in sig_chunks[0]["text"]

    # Verify every chunk has mandatory metadata
    for c in chunks:
        assert c["email_id"] == "enron_00001"
        assert c["chunk_id"].startswith("enron_00001_c")
        assert c["chunk_type"] in ("header_summary", "body_paragraph", "signature_block", "forwarded_thread", "list_block")
        assert len(c["chunk_reason"]) > 0
        assert c["source"] == "enron_corpus"
        assert len(c["text"]) > 0


def test_chunk_email_with_forward():
    chunks = chunk_email(SAMPLE_EMAIL_WITH_FORWARD, email_id="enron_00002")
    assert len(chunks) >= 3

    types = [c["chunk_type"] for c in chunks]
    assert "header_summary" in types
    assert "forwarded_thread" in types or any("forwarded" in c["chunk_reason"] for c in chunks)


if __name__ == "__main__":
    test_chunk_email_structure()
    test_chunk_email_with_forward()
    print("ALL CHUNKER TESTS PASSED!")
