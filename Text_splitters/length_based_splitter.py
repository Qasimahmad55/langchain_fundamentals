from langchain_text_splitters import CharacterTextSplitter

text = """=== PROJECT DISPATCH LOG ===
ID: 1042; Status: Pending; Priority: High; Region: North America
Notes: The client requested an immediate update on the database migration. 
We need to ensure all schema changes are backward-compatible.

ID: 1043; Status: Active; Priority: Medium; Region: Europe
Notes: Front-end team is waiting on the finalized API documentation. 
Deployment is tentatively scheduled for next Thursday.

ID: 1044; Status: Completed; Priority: Low; Region: Asia-Pacific
Notes: Server migration went smoothly with zero downtime reported.
Post-launch monitoring is now complete.
============================

Core Tasks for Next Sprint:
- Fix token expiration bug in auth service
- Optimize heavy SQL queries on reporting dashboard
- Update documentation for webhook integrations
"""

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="")

print(splitter.split_text(text))
