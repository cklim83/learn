Learning Resource: https://learngitbranching.js.org/

#### Starting A Repository
**git init**  - initialize a local repository in pwd. .git file will be created <br>
**git status** - check status of git repository (e.g. new files added, uncommitted changes to existing files) <br>
**git add [filename | .]**  - Stage files for commit. Can add filenames explicitly or using . for all modified files in current directory <br>

**git clone [repository address]**  - clone remote repository to pwd <br>

**git branch [branch_name]**  - create a pointer for new branch <br>
**git checkout [branch_name]** - switch current active branch to branch_name <br>
**git checkout -b [branch_name]** - shortcut to combine above two steps <br>

#### Merging Branches
**git merge [bugFix]**  - Merge bugFix into checkout branch. Assume we're on 'master' branch, merge content on bugFix to master <br>
**git rebase [branchname]** - Merge checkout branch into [branchname]. Assume we're on 'bugFix' branch, git rebase [branchname] will transfer all commits in bugFix not on [branchname] to [branchname] to form linear sequence. <br>

#### Moving Around Branch/HEAD Pointers
Note: HEAD is a symbolic name that points to current checkout commit. By default, it is hidden under the current checkout branch. HEAD can be detached from current checkout branch using absolute or relative reference. Location of HEAD is where the next commit will form from. <br>
**git log**  - List the hashes of the tree structure of local repository <br>
**git checkout commit_hash** - Shift HEAD to point at commit_hash using *absolute reference*. This is the only way to move forward i.e. child commit <br>
**git checkout master^** - Shift HEAD pointer to parent of commit pointed by "master" branch pointer using *relative reference* (relative to "master" location). <br>
**git checkout HEAD^** - shift HEAD to the parent(^) of the commit it is currently pointing <br>
**git checkout HEAD^^** - shift HEAD to the grandparent (^^) of the commit it is currently pointing <br>
**git checkout HEAD~n** - shift HEAD **n** commits back (~n) from the commit it is current pointing <br>

#### Move Branch Pointers i.e. Branch Forcing
**git branch -f branch_name [commit_hash]** - Moves (by force) branch_name pointer to commit with hash commit_hash. Move by absolute reference <br>
**git branch -f branch_name HEAD~n** - moves (by force) branch_name pointer to n parents before HEAD. HEAD is relative reference <br>

#### Reversing Change in Git Using Reset and Revert
**git reset HEAD~1** - reverts changes by moving **checkout branch reference** backwards in time by 1 commit. Reset works only for local but not remote repository.
**git revert HEAD** - reverts change by a adding a new commit that reverse changes in the current HEAD. In this way, the current in the current HEAD is negated when pulled by collaborators of the remote repository.<br>


#### Moving Work Around
**git cherry-pick [commit-1] [commit-2]** - copy a series of commits below your current location (HEAD) <br>
**git rebase -i target_location** - open a window to interactive select the commits and their order to be grown from target_location


#### Clone Remote Repository to Local
**git clone [address of remote repository]** - create a local copy of the remote repository. This process also automatically create a local remote branch called origin/master which tracks the status of the remote repository. Local remote branches have the special property that when you check them out, you are put into detached HEAD mode. Git does this on purpose because you can't work on these branches directly; you have to work elsewhere and then share your work with the remote (after which your remote branches will be updated.) origin/master only updates when remote repository updates. <br>

#### Git Fetch
**git fetch** - Download commits present in remote repository but missing locally, then update local remote branch pointer (e.g. origin/master or any origin/branch_name) to point to the commit pointed to at the remote branch. It does not change anything about your local state. It will not update your master branch or change anything about how your file system looks right now. <br>

**git pull** - Combines a git fetch and git merge to combine our checkout branch with the checkout branch in remote repository. Commonly known as pull request. <br>
**git pull --rebase** - Combines a git fetch, rebase current local commit onto the origin/master branch. <br>
**git push** - Publish new commits to remote repository and update our local origin/master pointer accordingly. Takes place after we synchronised our work with remote using git pull (i.e. git fetch + git merge/rebase/cherry-pick). Behavior of git push with no arguments varies depending on one of git's settings called push.default. The default value for this setting depends on the version of git you're using, but we are going to use the upstream value in our lessons. This isn't a huge deal, but it's worth checking your settings before pushing in your own projects. <br>
