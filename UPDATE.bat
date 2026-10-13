@echo off
rem Update Strata without starting the model (#475): the newest code (git pull, when this folder is a git clone), then
rem what START-HERE.bat does before a start - the engine (a new one when this version needs it), the Python packages,
rem each installed model's settings and draft subset. The model files are not touched. Start the model later with
rem START-HERE.bat. Options are passed on to setup.py.
setlocal
title Strata - update
cd /d "%~dp0"
rem This fork's own changes live on the dev branch (main is the upstream mirror): another branch would update
rem upstream's code and, with --video, build against the wrong tree. Say so instead of touching it, unless
rem STRATA_UPDATE_ANY_BRANCH is set. No pipe and no || chain here on purpose: cmd parses those inside a block
rem unreliably, and the first version (a piped findstr) refused a clone that was already on dev.
if defined STRATA_UPDATE_ANY_BRANCH goto :strata_branch_check_done
set "STRATA_BRANCH="
for /f "delims=" %%i in ('git symbolic-ref -q --short HEAD 2^>nul') do set "STRATA_BRANCH=%%i"
if not defined STRATA_BRANCH goto :strata_branch_check_done
if /i "%STRATA_BRANCH%"=="dev" goto :strata_branch_check_done
git show-ref --verify --quiet refs/remotes/origin/dev
if errorlevel 1 goto :strata_branch_check_done
echo  This is the video fork: its own changes live on the dev branch, and this clone is on "%STRATA_BRANCH%".
echo  Nothing was updated. Switch first, then run UPDATE.bat again:
echo    git switch dev
echo  Or set STRATA_UPDATE_ANY_BRANCH=1 to update this branch as it is.
exit /b 1

:strata_branch_check_done
rem Video updates deliberately avoid the legacy history-reset fallback below.
rem This mode accepts no setup options; use BUILD-VIDEO.ps1 directly for build options.
if /i "%~1"=="--video" (
  if not "%~2"=="" (
    echo  Usage: UPDATE.bat --video
    exit /b 1
  )
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD-VIDEO.ps1" -Update
  if errorlevel 1 exit /b 1
  exit /b 0
)
rem All of it in one block: cmd reads a .bat file while it runs it, and the git pull can change this very file.
(
  if exist ".git" (
    where git >nul 2>nul || (
      echo  This folder is a git clone, but git is not on PATH: install Git for Windows ^(winget install Git.Git^)
      echo  or run "git pull" here yourself, then run UPDATE.bat again.
      pause
      exit /b 1
    )
    echo  Getting the newest Strata ^(git pull^) ...
    git pull --ff-only
    if errorlevel 1 (
      rem The history was rewritten on 2026-10-06, see #1276: a clone from before has no commit in common with origin/main.
      set "REWR="
      git merge-base HEAD origin/main >nul 2>nul
      if errorlevel 1 set "REWR=1"
      git rev-parse --is-shallow-repository 2>nul | findstr /x true >nul && set "REWR="
      if defined REWR (
        echo.
        echo  This clone is on the repository's old history ^(cleaned up on 2026-10-06^): it has no commit in common
        echo  with origin/main, so it cannot be updated by git pull.
        set "DIRTY="
        for /f "delims=" %%i in ('git status --porcelain --untracked-files^=no') do set "DIRTY=1"
        if defined DIRTY (
          echo  You have local changes to tracked files here, so nothing was touched. To move to the new history
          echo  yourself ^(your untracked files, models and settings stay where they are^):
          echo    git branch pre-cleanup-backup
          echo    git stash push
          echo    git checkout -B main origin/main
          echo  then run UPDATE.bat again.
          pause
          exit /b 1
        )
        set "BK=pre-cleanup-backup"
        git rev-parse --verify -q refs/heads/pre-cleanup-backup >nul 2>nul && set "BK=pre-cleanup-backup-%RANDOM%"
        set "BR=main"
        for /f "delims=" %%i in ('git symbolic-ref --short -q HEAD 2^>nul') do set "BR=%%i"
        call git branch %%BK%% HEAD
        if errorlevel 1 (
          echo  Could not make the backup branch: nothing was updated.
          pause
          exit /b 1
        )
        call git checkout -q -B %%BR%% origin/main
        if errorlevel 1 (
          call git branch -D %%BK%% >nul 2>nul
          echo.
          echo  Could not move to the new history ^(the reason is above^): nothing was updated. By hand:
          echo    git checkout -B main origin/main
          pause
          exit /b 1
        )
        call echo  Moved to the new history. Your old commits are kept in the branch %%BK%%.
        echo  Untracked files ^(models, settings, the engine^) were not touched.
      ) else (
        echo.
        echo  git pull did not succeed ^(the reason is above^): nothing was updated. Files you changed here can stop it:
        echo  "git status" lists them.
        pause
        exit /b 1
      )
    )
  ) else (
    echo  This copy of Strata was not made with git, so it cannot fetch new files itself. Download the newest one:
    echo    https://github.com/msola-ht/Strata/archive/refs/heads/dev.zip
    echo  unzip it anywhere and run START-HERE.bat ^(or UPDATE.bat^) in it: it finds the model files in Strata-data
    echo  and sets itself up the same way - nothing big is downloaded again.
    echo  Checking this copy's engine and settings meanwhile ...
  )
  call "%~dp0START-HERE.bat" --update %*
  if not errorlevel 1 pause
  exit /b
)
