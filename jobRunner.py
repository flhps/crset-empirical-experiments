import yaml
import src.jobs.cascadeTest
import src.jobs.benchStatusCascade


JOB_DICT = {
    "cascadeTest": src.jobs.cascadeTest.run,
    "benchStatusCascade": src.jobs.benchStatusCascade.run,
}


def load_config(config_file="jobs.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

if __name__ == "__main__":
    # may be useful for future to also implement running jobs in parallel
    raw_jobs = CONFIG["sequence"]

    # preprocess the job list to expand matrix jobs
    jobs = []
    for raw in raw_jobs:
        if "repeat" in raw:
            for i in range(int(raw["repeat"])):
                j = raw.copy()
                del j["repeat"]
                jobs.append(j)
        else:
            jobs.append(raw.copy())

    print("Executing %s jobs in sequence" % len(jobs))

    for i in range(len(jobs)):
        job = jobs[i]
        job_type = job["type"]
        print("--- JOB %s ---" % i)
        print("Starting job: %s" % job_type)
        result = None
        if job_type not in JOB_DICT:
            print("UNKNOWN JOB TYPE")
            continue
        run_fun = JOB_DICT[job_type]
        result = run_fun(job["params"])
        print(result["message"])
        # todo save a csv with all the values?
