@echo off

rem ---------------------------------------------------------------------
rem Script: capture_and_organize_screenshots.bat
rem Author: Mohamed Tag
rem Date: 2024-3-12
rem Purpose: This script captures screenshots from an Android device, 
rem          organizes them by category, and pulls them to a local folder 
rem          on your computer. It is intended for use in collecting data 
rem          for image classification projects.
rem ---------------------------------------------------------------------

rem Set the path to the Android storage where screenshots will be saved
set android_path="/storage/emulated/0/Download/Slash"

rem Set the local folder path where screenshots will be pulled and categorized
set local_folder="C:\my-files\personal-projects\slash-image-classification\Data collection\screenshots"

rem Initial screenshot number
set screenshot_number=1

rem Capture loop to take screenshots and organize them by category
:capture_loop
rem Wait for a key press to capture a new screenshot
pause

rem Capture screenshot on the device
adb shell screencap -p %android_path%/screenshot_%screenshot_number%.png

rem Pull the screenshot to your computer
adb pull %android_path%/screenshot_%screenshot_number%.png %local_folder%

rem Increment screenshot number
set /a screenshot_number+=1

rem Go back to the beginning of the loop
goto capture_loop
