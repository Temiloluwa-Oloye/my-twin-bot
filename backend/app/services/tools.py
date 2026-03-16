import httpx

async def get_latest_github_commits(username: str) -> str:
    """
    Fetches the latest public commits for a specific GitHub user.
    This function will be triggered by the LLM when asked about recent coding activity.
    """
    url = f"https://api.github.com/users/{username}/events/public"
    
    async with httpx.AsyncClient() as client:
        # Using a low timeout so the bot doesn't hang if GitHub is slow
        response = await client.get(url, timeout=5.0)
        
    if response.status_code != 200:
        return f"Could not fetch GitHub data. Status: {response.status_code}"
        
    events = response.json()
    # Filter for actual code pushes, grab the 3 most recent
    push_events = [e for e in events if e.get("type") == "PushEvent"][:3]
    
    if not push_events:
        return f"{username} has no recent public commits."
        
    result = f"Latest public GitHub activity for {username}:\n"
    for event in push_events:
        repo_name = event["repo"]["name"]
        commits = event["payload"].get("commits", [])
        for commit in commits:
            # Clean up the commit message to prevent massive prompt injections
            message = commit['message'].split('\n')[0][:100] 
            result += f"- Repository: {repo_name} | Commit: {message}\n"
            
    return result

# We define the strict JSON schema that Groq requires to understand our tool
GITHUB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_latest_github_commits",
        "description": "Fetch the latest public GitHub commits and coding activity for Temi. Use this ONLY when the user asks what Temi is currently coding, building, or working on recently.",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "The GitHub username to look up."
                }
            },
            "required": ["username"]
        }
    }
}