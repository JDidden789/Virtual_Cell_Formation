
import random
import itertools
import warnings
warnings.filterwarnings('ignore')

def get_iat(operations_per_job, demand, operation_demand, processingTime, batch_size, no_machine, utilization):
    import numpy as np
    mean_operations = np.average(operations_per_job, weights=demand)
    mean_processing_time = np.average(processingTime, weights=operation_demand)
    mean_batch = np.average(batch_size, weights=operation_demand)

    return mean_operations*mean_processing_time*mean_batch / (no_machine * utilization)

def generate_instance(no_jobs, no_tools, no_fixtures, no_machines, h, batch_size, flex, iter, demand, machine_types):
    from scipy.stats import truncnorm
    import numpy as np
    np.random.seed(iter)
    random.seed(iter)

    operations_per_job = np.random.choice(range(1, 5), size=no_jobs)
    myclip_a = 0
    myclip_b = 5 * 60
    my_mean = 40
    my_std = 30

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    processingTime = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=sum(operations_per_job))

    operations_numbering = []
    k = 0
    for i in range(no_jobs):
        op = []
        for _ in range(operations_per_job[i]):
            op.append(k)
            k+= 1
        operations_numbering.append(op)

# Generate batch size
    batch_operation = []
    batch = np.random.choice(range(1, 2), sum(operations_per_job))
    for i in range(no_jobs):
        batch_operation.extend(list(itertools.repeat(batch[i], operations_per_job[i])))
# Create Demand Matrix
    demand_operation = []

    for i in range(no_jobs):
        demand_operation.extend(list(itertools.repeat(demand[i], operations_per_job[i])))

    iat = get_iat(operations_per_job, demand, demand_operation, processingTime, batch, no_machines, 0.9)

    demand_o = np.zeros(sum(operations_per_job))
    for o in range(sum(operations_per_job)):
        demand_o[o] = h * demand_operation[o]/iat*processingTime[o]*batch_operation[o]
# Create Tool Matrix
    tools = np.zeros((sum(operations_per_job), no_tools))
    for i in range(sum(operations_per_job)):
        tool_choice = np.random.choice(range(no_tools), random.randint(1, 1))
        tools[i, tool_choice] = 1
# Create Fixture Matrix
    fixtures = np.zeros((sum(operations_per_job), no_fixtures))
    weights = np.concatenate(([0.8], np.repeat(0.2 / (no_fixtures - 1), no_fixtures - 1)), axis=None)
    for i in range(sum(operations_per_job)):
        fixture_choice = np.random.choice(range(no_fixtures), random.randint(1, 1), p=weights)
        fixtures[i, fixture_choice] = 1
# Create machine matrix
    operation_routing = []
    for _ in range(sum(operations_per_job)):
        size = random.choice(range(1, 4))
        operation_routing.append(random.sample(range(0, machine_types), k=size))

    machinestype = np.zeros((machine_types, sum(operations_per_job)))
    for o in range(sum(operations_per_job)):
        for k in operation_routing[o]:
            machinestype[k][o] = 1

    machine_capacity = np.zeros(machine_types)
    for o in range(sum(operations_per_job)):
        for k in operation_routing[o]:
            machine_capacity[k] += demand_o[o] / len(operation_routing[o])

    machines_per_type = [int(np.ceil(machine_capacity[k] / sum(machine_capacity) * no_machines)) for k in range(machine_types)]


    machines = []
    for m in range(machine_types):
        for _ in range(machines_per_type[m]):
            machines.append(list(machinestype[m]))

    machines = [list(i) for i in zip(*machines)]

# Create tool-machine matrix
    tool_machine = np.zeros((no_tools, no_machines))
    for i in range(sum(operations_per_job)):
        tool_comp = np.where(tools[i] == 1)
        machine_comp = np.where(machines[i] == 1)
        tool_machine[tool_comp[0], machine_comp[0]] = 1

# Create fixture-machine matrix
    fixture_machine = np.zeros((no_fixtures, no_machines))
    for i in range(sum(operations_per_job)):
        fixture_comp = np.where(fixtures[i] == 1)
        machine_comp = np.where(machines[i] == 1)
        fixture_machine[fixture_comp[0], machine_comp[0]] = 1


# Create Setup Matrix
    max_setup = 60
    b = np.random.uniform(15, max_setup, size=(no_tools, no_tools))
    setupTime = (b + b.T) / 2
    for i in range(no_tools):
        setupTime[i][i] = 0

    setupTime_new = np.zeros((sum(operations_per_job), sum(operations_per_job)))

    for i in range(sum(operations_per_job)):
        for j in range(1, sum(operations_per_job)):
            tool1 = np.where(tools[i] == 1)
            tool2 = np.where(tools[j] == 1)
            setupTime_new[i][j] = setupTime[tool1[0][0]][tool2[0][0]]

    machine_position = random.sample(range(100), no_machines)
    absolute_machine_position = [
        (int(np.floor(i / 10)), i % 10) for i in machine_position
    ]
    distance = np.zeros((no_machines, no_machines))
    for i in range(no_machines):
        for j in range(no_machines):
            distance[i, j] = (np.abs(absolute_machine_position[i][0] - absolute_machine_position[j][0]) * 5 + np.abs(absolute_machine_position[i][1] - absolute_machine_position[j][1]) * 5) / 100
    return operations_per_job, operations_numbering, demand, demand_o, tools, fixtures, machines, tool_machine, fixture_machine, setupTime_new, distance

def jaccard_binary(x,y):
    import numpy as np
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    return intersection.sum() / float(union.sum())

def get_similiarity(no_jobs, operations_per_job, demand, tools, fixtures, machines, setupTime_new, weights):
    import numpy as np
    tool_sim = np.zeros((sum(operations_per_job), sum(operations_per_job)))
    for i in range(sum(operations_per_job) - 1):
        for j in range(1, sum(operations_per_job)):
            tool_sim[i, j] = jaccard_binary(tools[i], tools[j])

    fixture_sim = np.zeros((sum(operations_per_job), sum(operations_per_job)))
    for i in range(sum(operations_per_job) - 1):
        for j in range(1, sum(operations_per_job)):
            fixture_sim[i, j] = jaccard_binary(fixtures[i], fixtures[j])

    machine_sim = np.zeros((sum(operations_per_job), sum(operations_per_job)))
    for i in range(sum(operations_per_job) - 1):
        for j in range(1, sum(operations_per_job)):
            machine_sim[i, j] = jaccard_binary(machines[i], machines[j])
            # print(machine_sim[i, j], machines[i], machines[j])
            if np.isnan(machine_sim[i,j]):
                print(machines[i], machines[j])

    demand_sim = np.zeros((sum(operations_per_job), sum(operations_per_job)))
    for i, j in itertools.product(range(no_jobs), range(1, no_jobs)):
        demand_sim[i, j] = 1 - (0.5 * np.abs(demand[i] - demand[j]) / (max(demand) - min(demand)) + 0.5 * np.abs(demand[i] - demand[j]) / (max(demand[i], demand[j])))

    setup_sim = np.ones((sum(operations_per_job), sum(operations_per_job)))
    for k in range(sum(operations_per_job)):
        for m in range(sum(operations_per_job)):
            setup_sim[k, m] = 1 - setupTime_new[k][m] / (np.max(setupTime_new) - np.min(setupTime_new))
            setup_sim[m, k] = 1 - setupTime_new[k][m] / (np.max(setupTime_new) - np.min(setupTime_new))


    similiarty_final = np.ones((sum(operations_per_job), sum(operations_per_job)))
    for i in range(sum(operations_per_job)):
        for j in range(1, sum(operations_per_job)):

            similiarty_final[i, j] = weights[0]*tool_sim[i, j] + weights[1]*fixture_sim[i, j]+weights[2]*machine_sim[i, j]+weights[3]*(demand_sim[i,j])+weights[4]*setup_sim[i, j]
            similiarty_final[j, i] = weights[0]*tool_sim[i, j] + weights[1]*fixture_sim[i, j]+weights[2]*machine_sim[i, j]+weights[3]*(demand_sim[i,j])+weights[4]*setup_sim[i, j]
    return similiarty_final

def generate_instance_learning(no_jobs, no_tools, no_fixtures, no_machines, h, batch_size, flex, iter, demand, machine_types):
    from scipy.stats import truncnorm
    import numpy as np
    np.random.seed(iter)
    random.seed(iter)

    operations_per_job = np.random.choice(range(1, 5), size=no_jobs)
    myclip_a = 0
    myclip_b = 5 * 60
    my_mean = 91
    my_std = 51

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    processingTime = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=sum(operations_per_job))

    operations_numbering = []
    k = 0
    for i in range(no_jobs):
        op = []
        for _ in range(operations_per_job[i]):
            op.append(k)
            k+= 1
        operations_numbering.append(op)

# Generate batch size
    batch_operation = []
    batch = np.random.choice(range(1, 2), sum(operations_per_job))
    for i in range(no_jobs):
        batch_operation.extend(list(itertools.repeat(batch[i], operations_per_job[i])))
# Create Demand Matrix
    # demand = np.random.choice(range(10, 100), no_jobs)
    # demand = [x / sum(demand) for x in demand]
    demand_operation = []

    for i in range(no_jobs):
        demand_operation.extend(list(itertools.repeat(demand[i], operations_per_job[i])))

    iat = get_iat(operations_per_job, demand, demand_operation, processingTime, batch, no_machines, 0.85)

    demand_o = np.zeros(sum(operations_per_job))
    for o in range(sum(operations_per_job)):
        demand_o[o] = h * demand_operation[o]/iat*processingTime[o]*batch_operation[o]
# Create Tool Matrix
    tools = np.zeros((sum(operations_per_job), no_tools))
    for i in range(sum(operations_per_job)):
        tool_choice = np.random.choice(range(no_tools), random.randint(1, 1))
        tools[i, tool_choice] = 1
# Create Fixture Matrix
    fixtures = np.zeros((sum(operations_per_job), no_fixtures))
    weights = np.concatenate(([0.8], np.repeat(0.2 / (no_fixtures - 1), no_fixtures - 1)), axis=None)
    for i in range(sum(operations_per_job)):
        fixture_choice = np.random.choice(range(no_fixtures), random.randint(1, 1), p=weights)
        fixtures[i, fixture_choice] = 1
# Create machine matrix
    operation_routing = []
    for _ in range(sum(operations_per_job)):
        size = random.choice(range(1, 4))
        operation_routing.append(random.sample(range(0, machine_types), k=size))

    machinestype = np.zeros((machine_types, sum(operations_per_job)))
    for o in range(sum(operations_per_job)):
        for k in operation_routing[o]:
            machinestype[k][o] = 1

    machine_capacity = np.zeros(machine_types)
    for o in range(sum(operations_per_job)):
        for k in operation_routing[o]:
            machine_capacity[k] += demand_o[o] / len(operation_routing[o])

    machines_per_type = [int(np.ceil(machine_capacity[k] / sum(machine_capacity) * no_machines)) for k in range(machine_types)]


    machines = []
    for m in range(machine_types):
        for _ in range(machines_per_type[m]):
            machines.append(list(machinestype[m]))

    machines = [list(i) for i in zip(*machines)]

    print(sum(machines_per_type))

# Create tool-machine matrix
    tool_machine = np.zeros((no_tools, no_machines))
    for i in range(sum(operations_per_job)):
        tool_comp = np.where(tools[i] == 1)
        machine_comp = np.where(machines[i] == 1)
        tool_machine[tool_comp[0], machine_comp[0]] = 1

# Create fixture-machine matrix
    fixture_machine = np.zeros((no_fixtures, no_machines))
    for i in range(sum(operations_per_job)):
        fixture_comp = np.where(fixtures[i] == 1)
        machine_comp = np.where(machines[i] == 1)
        fixture_machine[fixture_comp[0], machine_comp[0]] = 1

    proccesingTimeNew = []
    k=0
    for i in range(no_jobs):
        local_proc = []
        for j in operations_numbering[i]:
            local_proc.append(processingTime[k] * batch[i])
            k +=1
        proccesingTimeNew.append(local_proc)
    # print(max(max(proccesingTimeNew)))
    # print(np.sum(proccesingTimeNew, axis=1))

    sumProcessingTime = [sum(proccesingTimeNew[i]) * batch[i] for i in range(no_jobs)]
    # print(sumProcessingTime)
    # Get maximum critical ratio
    max_ddt = 8
    CR = []
    DDT = []
    for j in range(no_jobs):
        CR.append(
            [
                (sum(proccesingTimeNew[j]) * max_ddt - sum(proccesingTimeNew[j][:i])) / sum(proccesingTimeNew[j][i:])
                for i in range(operations_per_job[j] - 1)
            ]
        )
        DDT.append(sum(proccesingTimeNew[j]) * max_ddt)


# Create Setup Matrix
    max_setup = 60
    b = np.random.uniform(15, max_setup, size=(no_tools, no_tools))
    setupTime = (b + b.T) / 2
    for i in range(no_tools):
        setupTime[i][i] = 0

    setupTime_new = np.zeros((sum(operations_per_job), sum(operations_per_job)))

    for i in range(sum(operations_per_job)):
        for j in range(1, sum(operations_per_job)):
            tool1 = np.where(tools[i] == 1)
            tool2 = np.where(tools[j] == 1)
            setupTime_new[i][j] = setupTime[tool1[0][0]][tool2[0][0]]

    machine_position = random.sample(range(100), no_machines)
    absolute_machine_position = [
        (int(np.floor(i / 10)), i % 10) for i in machine_position
    ]
    distance = np.zeros((no_machines, no_machines))
    for i in range(no_machines):
        for j in range(no_machines):
            distance[i, j] = (np.abs(absolute_machine_position[i][0] - absolute_machine_position[j][0]) * 5 + np.abs(absolute_machine_position[i][1] - absolute_machine_position[j][1]) * 5) / 100
    return processingTime, operations_numbering, setupTime_new, demand, iat, sum(operations_per_job), batch, machines, tools, fixtures, machine_position, CR, DDT, distance, max(sumProcessingTime), max(proccesingTimeNew)[0], demand_operation
