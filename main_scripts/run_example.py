#!/usr/bin/env python3

import os, json, subprocess

scripts_dir = "py_scripts"
exec_command = "python3 {sd}/template.py".format(sd=scripts_dir)
training_root = os.getenv("TRAINING_ROOT")
e2e_root = os.getenv("E2E_ROOT")

settings = None
with open("{er}/settings.json".format(er=e2e_root)) as settings_file_handle:
    settings = json.load(settings_file_handle)

exec_command += " --test {v}".format(v=settings["test"]["example"])

log_dir = "logs"
log_file_name_stdout = "example_stdout.log"
log_file_name_stderr = "example_stderr.log"
with open("{ld}/{lf}".format(ld=log_dir, lf=log_file_name_stdout), 'w') as log_stdout_handle, open("{ld}/{lf}".format(ld=log_dir, lf=log_file_name_stderr), 'w') as log_stderr_handle:
    subprocess.check_call("cd {tr} && {ec}".format(tr=training_root, ec=exec_command), shell=True, executable="/bin/bash", stdout=log_stdout_handle, stderr=log_stderr_handle)
