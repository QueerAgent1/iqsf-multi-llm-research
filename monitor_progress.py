#!/usr/bin/env python3
"""
IQSF Publication Progress Monitor
Real-time tracking of multi-LLM publication generation
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

def check_system_status():
    """Check overall system status"""
    try:
        with open('/home/ubuntu/iqsf-multi-llm-research/system_status.json', 'r') as f:
            status = json.load(f)
        return status
    except:
        return {"status": "not_initialized"}

def check_workspace_activity():
    """Check activity in each workspace"""
    workspaces = [
        "literature", "methodology", "analysis", 
        "writing", "review", "submission"
    ]
    
    activity = {}
    base_path = "/home/ubuntu/iqsf-multi-llm-research/workspaces"
    
    for workspace in workspaces:
        workspace_path = f"{base_path}/{workspace}"
        if os.path.exists(workspace_path):
            # Count files in each subdirectory
            subdirs = ["inputs", "outputs", "drafts", "reviews", "final"]
            counts = {}
            for subdir in subdirs:
                subdir_path = f"{workspace_path}/{subdir}"
                if os.path.exists(subdir_path):
                    counts[subdir] = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                else:
                    counts[subdir] = 0
            activity[workspace] = counts
        else:
            activity[workspace] = {"status": "not_found"}
    
    return activity

def check_paper_progress():
    """Check progress on each foundation paper"""
    try:
        with open('/home/ubuntu/iqsf-multi-llm-research/foundation_papers_config.json', 'r') as f:
            config = json.load(f)
        
        papers = config.get('foundation_papers', [])
        progress = {}
        
        for paper in papers:
            paper_id = paper['paper_id']
            title = paper['title']
            target = paper['target_journal']
            
            # Check for paper-specific files
            paper_files = {
                'literature_review': f"/home/ubuntu/iqsf-multi-llm-research/workspaces/literature/outputs/{paper_id}_literature.md",
                'methodology': f"/home/ubuntu/iqsf-multi-llm-research/workspaces/methodology/outputs/{paper_id}_methodology.md",
                'analysis': f"/home/ubuntu/iqsf-multi-llm-research/workspaces/analysis/outputs/{paper_id}_analysis.md",
                'draft': f"/home/ubuntu/iqsf-multi-llm-research/workspaces/writing/outputs/{paper_id}_draft.md",
                'review': f"/home/ubuntu/iqsf-multi-llm-research/workspaces/review/outputs/{paper_id}_review.md",
                'final': f"/home/ubuntu/iqsf-multi-llm-research/publications/{paper_id}_final.pdf"
            }
            
            completion = {}
            for stage, file_path in paper_files.items():
                completion[stage] = os.path.exists(file_path)
            
            progress[paper_id] = {
                'title': title,
                'target_journal': target,
                'completion': completion,
                'progress_percentage': sum(completion.values()) / len(completion) * 100
            }
        
        return progress
    except:
        return {}

def check_agent_activity():
    """Check LLM agent activity logs"""
    agent_logs = {}
    log_path = "/home/ubuntu/iqsf-multi-llm-research/outputs"
    
    if os.path.exists(log_path):
        for file in os.listdir(log_path):
            if file.endswith('.log') or file.endswith('.json'):
                file_path = os.path.join(log_path, file)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    agent_logs[file] = {
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'exists': True
                    }
    
    return agent_logs

def display_progress():
    """Display comprehensive progress dashboard"""
    
    print("\n" + "="*80)
    print("🚀 IQSF MULTI-LLM PUBLICATION SYSTEM - PROGRESS MONITOR")
    print("="*80)
    print(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System Status
    print("\n📊 SYSTEM STATUS:")
    status = check_system_status()
    if 'foundation_papers' in status:
        print(f"   ✅ System Initialized: {status.get('initialization_time', 'Unknown')}")
        print(f"   📚 Foundation Papers: {status.get('foundation_papers', 0)}")
        print(f"   🤖 LLM Agents: {status.get('llm_agents', 0)}")
        print(f"   🏗️ Workspaces: {status.get('workspaces_created', 0)}")
        print(f"   👤 Primary Author: {status.get('author', 'Unknown')}")
    else:
        print("   ❌ System not properly initialized")
    
    # Workspace Activity
    print("\n🏗️ WORKSPACE ACTIVITY:")
    activity = check_workspace_activity()
    for workspace, counts in activity.items():
        if isinstance(counts, dict) and 'status' not in counts:
            total_files = sum(counts.values())
            print(f"   📁 {workspace.title()}: {total_files} files")
            if total_files > 0:
                for subdir, count in counts.items():
                    if count > 0:
                        print(f"      └── {subdir}: {count} files")
        else:
            print(f"   📁 {workspace.title()}: No activity")
    
    # Paper Progress
    print("\n📄 FOUNDATION PAPERS PROGRESS:")
    papers = check_paper_progress()
    if papers:
        for paper_id, info in papers.items():
            progress_pct = info['progress_percentage']
            title = info['title'][:60] + "..." if len(info['title']) > 60 else info['title']
            target = info['target_journal']
            
            # Progress bar
            bar_length = 20
            filled_length = int(bar_length * progress_pct / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            print(f"   📋 {title}")
            print(f"      Target: {target}")
            print(f"      Progress: [{bar}] {progress_pct:.1f}%")
            
            # Show completed stages
            completed_stages = [stage for stage, done in info['completion'].items() if done]
            if completed_stages:
                print(f"      Completed: {', '.join(completed_stages)}")
            print()
    else:
        print("   ❌ No paper progress data found")
    
    # Agent Activity
    print("\n🤖 AGENT ACTIVITY LOGS:")
    logs = check_agent_activity()
    if logs:
        for log_file, info in logs.items():
            size_kb = info['size'] / 1024
            modified = info['modified']
            print(f"   📝 {log_file}: {size_kb:.1f} KB (modified: {modified})")
    else:
        print("   ❌ No agent activity logs found")
    
    # Next Steps
    print("\n🔧 NEXT STEPS:")
    if not papers:
        print("   • Initialize publication generation system")
        print("   • Configure LLM API credentials")
        print("   • Start literature review process")
    else:
        in_progress = [p for p in papers.values() if 0 < p['progress_percentage'] < 100]
        completed = [p for p in papers.values() if p['progress_percentage'] == 100]
        
        if completed:
            print(f"   ✅ {len(completed)} papers completed")
        if in_progress:
            print(f"   🔄 {len(in_progress)} papers in progress")
        if len(completed) + len(in_progress) < len(papers):
            print(f"   ⏳ {len(papers) - len(completed) - len(in_progress)} papers pending")
    
    print("\n" + "="*80)

def monitor_continuously():
    """Continuously monitor progress"""
    try:
        while True:
            os.system('clear')  # Clear screen
            display_progress()
            print("\n⏱️  Refreshing in 10 seconds... (Ctrl+C to exit)")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuously()
    else:
        display_progress()
        print("\n💡 Run with --continuous flag for real-time monitoring")

