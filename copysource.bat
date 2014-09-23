echo "Copy from repository to running folder"
robocopy "C:\cygwin\home\mramire8\python_code\sr\active" "." *.py /S /XO /V /XA:H /XD .git
echo "Done copying."

