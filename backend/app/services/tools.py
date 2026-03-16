import httpx

async def get_latest_github_commits(username: str) -> str:
    """
    Fetches the latest public commits for a specific GitHub user.
    """
    url = f"https://api.github.com/users/{username}/events/public"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            
        if response.status_code != 200:
            return f"Could not fetch GitHub data. Status: {response.status_code}"
            
        events = response.json()
        push_events = [e for e in events if e.get("type") == "PushEvent"][:3]
        
        if not push_events:
            return f"{username} has no recent public commits."
            
        result = f"Latest public GitHub activity for {username}:\n"
        result = f"Latest public GitHub activity for {username}:\n"
        for event in push_events:
            repo_name = event["repo"]["name"]
            commits = event["payload"].get("commits", [])
            
            # If the push event has no explicit commit messages, just list the repo
            if not commits:
                result += f"- Pushed updates to repository: {repo_name} (No commit details available)\n"
            
            # If it does have commit messages, list them out
            for commit in commits:
                message = commit.get("message", "Update").split('\n')[0][:100] 
                result += f"- Repository: {repo_name} | Commit: {message}\n"
                
        return result
        
    except Exception as e:
        # If your network blocks port 443, the bot will gracefully report it instead of crashing!
        return f"Network error when trying to reach GitHub: {str(e)}"

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