import sys
if (sys.version_info.major < 3): sys.exit("Must be using py3 onwards. Current version info: {v}".format(v=sys.version_info))
if (sys.version_info.minor < 6): sys.exit("Must be using python 3.6 onwards. Current version info: {v}".format(v=sys.version_info))

import os, subprocess

training_root = os.getenv("TRAINING_ROOT")
training_eos_root = os.getenv("TRAINING_EOS_ROOT")
training_archives = os.getenv("TRAINING_ARCHIVES")
training_analysisroot = os.getenv("TRAINING_ANALYSISROOT")
eos_prefix = os.getenv("EOSPREFIX")
tmUtils_parent = os.getenv("TM_UTILS_PARENT")
hostname = os.getenv("HOSTNAME")
x509Proxy = os.getenv("X509_USER_PROXY")
condor_work_area_root = os.getenv("CONDORWORKAREAROOT")
scratch_area = os.getenv("SCRATCHAREA")
habitat = ""
if ("lxplus" in hostname):
    habitat = "lxplus"
elif ("fnal" in hostname):
    habitat = "fnal"
else:
    sys.exit("ERROR: Unrecognized hostname: {h}, seems to be neither lxplus nor fnal.".format(h=hostname))

def get_execution_command(commandToRun):
    env_setup_command = "bash -c \"cd {tr}/sh_snippets && source setup_env.sh && cd ..".format(tr=training_root)
    return "{e_s_c} && set -x && {c} && set +x\"".format(e_s_c=env_setup_command, c=commandToRun)

def execute_in_env(commandToRun, isDryRun=False, functionToCallIfCommandExitsWithError=None):
    executionCommand = get_execution_command(commandToRun)
    if (isDryRun):
        print("Dry-run, not executing:")
        print("{c}".format(c=executionCommand))
    else:
        print("Executing:")
        print("{c}".format(c=executionCommand))
        try:
            subprocess.check_call(executionCommand, shell=True, executable="/bin/bash")
        except subprocess.CalledProcessError as error_handle:
            if not(functionToCallIfCommandExitsWithError is None):
                if not(callable(functionToCallIfCommandExitsWithError)): sys.exit("ERROR in execute_in_env: command exited with error and unable to call functionToCallIfCommandExitsWithError")
                else: functionToCallIfCommandExitsWithError()
            sys.exit("ERROR: command \"{c}\" failed with return code {r}.".format(c=commandToRun, r=error_handle.returncode))
