from kb import build_knowledgebase


if __name__ == "__main__":
    report = build_knowledgebase(
        input_dir="input_docs",
        output_dir="output"
    )

    print("\nClinical Knowledge Base Build Complete")
    print("-" * 45)

    print(
        "Documents discovered :",
        report["documents_discovered"]
    )

    print(
        "Documents processed  :",
        report["documents_processed"]
    )

    print(
        "Sections created     :",
        report["sections_created"]
    )

    print(
        "Chunks created       :",
        report["chunks_created"]
    )

    print(
        "Failed documents     :",
        report["failed_documents"]
    )

    print(
        "Avg chunk chars      :",
        report["average_chunk_chars"]
    )

    print(
        "Retrieval ready      :",
        "YES"
        if report["retrieval_ready"]
        else "CHECK REPORT"
    )

    if report["failures"]:
        print("\nFailures:")
        print("-" * 45)

        for failure in report["failures"]:
            print(
                f"- {failure['file']}: "
                f"{failure['error']}"
            )