import numpy as np

# The goal of this routine is to return the minimum cost dynamic programming
# solution given a set of unary and pairwise costs
def dynamicProgram(unaryCosts, pairwiseCosts):

    # Count number of positions (pixels in the scanline) and nodes at each position
    nNodesPerPosition = unaryCosts.shape[0]  # Number of disparities
    nPosition = unaryCosts.shape[1]          # Number of positions (pixels)

    # Initialize minimum cost matrix and parent matrix
    minimumCost = np.zeros((nNodesPerPosition, nPosition))
    parents = np.zeros((nNodesPerPosition, nPosition), dtype=int)

    # FORWARD PASS

    # Fill in the first column of the minimum cost matrix with unary costs
    minimumCost[:, 0] = unaryCosts[:, 0]

    # Iterate over each position starting from the second one
    for cPosition in range(1, nPosition):
        # Iterate over each node (disparity level) at the current position
        for cNode in range(nNodesPerPosition):
            # Compute costs from all possible previous nodes to the current node
            possPathCosts = np.zeros(nNodesPerPosition)
            for cPrevNode in range(nNodesPerPosition):
                # Total cost: previous minimum cost + pairwise cost + current unary cost
                cost = (minimumCost[cPrevNode, cPosition - 1] +
                        pairwiseCosts[cPrevNode, cNode] +
                        unaryCosts[cNode, cPosition])
                possPathCosts[cPrevNode] = cost

            # Find the minimum cost and the corresponding previous node
            minCost = np.min(possPathCosts)
            ind = np.argmin(possPathCosts)

            # Store the minimum cost and the parent index
            minimumCost[cNode, cPosition] = minCost
            parents[cNode, cPosition] = ind

    # BACKWARD PASS

    # Initialize the bestPath array
    bestPath = np.zeros(nPosition, dtype=int)

    # Find the node with the minimum cost at the last position
    minInd = np.argmin(minimumCost[:, nPosition - 1])
    bestPath[-1] = minInd

    # Trace back the best path using the parents matrix
    for cPosition in range(nPosition - 1, 0, -1):
        bestParent = parents[bestPath[cPosition], cPosition]
        bestPath[cPosition - 1] = bestParent

    return bestPath

def dynamicProgramVec(unaryCosts, pairwiseCosts):
    # Count number of positions (pixels in the scanline) and nodes at each position
    nNodesPerPosition, nPosition = unaryCosts.shape

    # Initialize minimum cost matrix and parent matrix
    minimumCost = np.zeros((nNodesPerPosition, nPosition))
    parents = np.zeros((nNodesPerPosition, nPosition), dtype=int)

    # FORWARD PASS

    # Fill in the first column of the minimum cost matrix with unary costs
    minimumCost[:, 0] = unaryCosts[:, 0]

    # Iterate over each position starting from the second one
    for cPosition in range(1, nPosition):
        # Previous minimum costs (from position cPosition - 1)
        prev_min_costs = minimumCost[:, cPosition - 1]

        # Compute the total costs for all possible transitions
        # Shape of total_costs: (nNodesPerPosition, nNodesPerPosition)
        total_costs = prev_min_costs[:, np.newaxis] + pairwiseCosts

        # Add the unary costs for the current position
        # The unaryCosts for position cPosition are added to each column
        total_costs += unaryCosts[:, cPosition]

        # Find the minimum cost and corresponding parent for each node at position cPosition
        min_costs = np.min(total_costs, axis=0)
        min_indices = np.argmin(total_costs, axis=0)

        # Store the minimum costs and parent indices
        minimumCost[:, cPosition] = min_costs
        parents[:, cPosition] = min_indices

    # BACKWARD PASS

    # Initialize the bestPath array
    bestPath = np.zeros(nPosition, dtype=int)

    # Find the node with the minimum cost at the last position
    minInd = np.argmin(minimumCost[:, nPosition - 1])
    bestPath[-1] = minInd

    # Trace back the best path using the parents matrix
    for cPosition in range(nPosition - 1, 0, -1):
        bestParent = parents[bestPath[cPosition], cPosition]
        bestPath[cPosition - 1] = bestParent

    return bestPath
