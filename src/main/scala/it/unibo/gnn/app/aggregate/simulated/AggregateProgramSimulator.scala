package it.unibo.gnn.app.aggregate.simulated

import it.unibo.gnn.app.aggregate.simulated.Network.NetworkWithOps
import it.unibo.gnn.model.GraphNeuralNetwork
import it.unibo.gnn.model.GraphNeuralNetwork.{NeighborhoodData, NodeState}
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * It simulates an aggregate program evaluation. It uses internally only the GNN interaction with the
 * neighbourhood.
 */
trait AggregateProgramSimulator {
  def evaluate(
    network: NetworkWithOps[AggregateProgramSimulator.ID],
    id: AggregateProgramSimulator.ID
  ): NetworkWithOps[AggregateProgramSimulator.ID]
  def bulkEvaluation(
    network: NetworkWithOps[AggregateProgramSimulator.ID],
    ids: Set[AggregateProgramSimulator.ID]
  ): NetworkWithOps[AggregateProgramSimulator.ID]
  def timeSeries(
    network: NetworkWithOps[AggregateProgramSimulator.ID],
    ids: Iterable[AggregateProgramSimulator.ID]
  ): NetworkWithOps[AggregateProgramSimulator.ID]
  def timeSeriesAndBulk(
     network: NetworkWithOps[AggregateProgramSimulator.ID],
     evaluation: Iterable[Set[AggregateProgramSimulator.ID]]
  ): NetworkWithOps[AggregateProgramSimulator.ID]
}

object AggregateProgramSimulator {
  type ID = Int
  def simulatorFromGNN(
    gnn: GraphNeuralNetwork,
    featureUpdate : (INDArray, INDArray) => INDArray
  ) : AggregateProgramSimulator = new AggregateProgramSimulator {
    override def evaluate(
      network: NetworkWithOps[AggregateProgramSimulator.ID],
      id: ID
    ): NetworkWithOps[AggregateProgramSimulator.ID] = {
      val result = graphNetworkEvaluation(network, id)
      network
        .updateState(id, result.state)
        .updateFeature(id, featureUpdate(network.feature(id), result.output))
        .updateOutput(id, result.output)
    }

    override def bulkEvaluation(
        network: NetworkWithOps[AggregateProgramSimulator.ID],
        ids: Set[ID]
      ): NetworkWithOps[AggregateProgramSimulator.ID] = {
      val results = ids.map(id => id -> graphNetworkEvaluation(network, id))
      results.foldLeft(network){
        case(acc, (id, result)) => acc
          .updateState(id, result.state)
          .updateFeature(id, featureUpdate(network.feature(id), result.output))
          .updateOutput(id, result.output)
      }
    }

    override def timeSeries(
        network: NetworkWithOps[AggregateProgramSimulator.ID],
        ids: Iterable[ID]
      ): NetworkWithOps[AggregateProgramSimulator.ID] = {
      ids.foldLeft(network){
        case (network, id) => evaluate(network, id)
      }
    }

    override def timeSeriesAndBulk(
      network: NetworkWithOps[AggregateProgramSimulator.ID],
      evaluations: Iterable[Set[ID]]
    ): NetworkWithOps[AggregateProgramSimulator.ID] = {
      evaluations.foldLeft(network){
        case (network, ids) => bulkEvaluation(network, ids)
      }
    }

    def graphNetworkEvaluation(network: Network[ID], id: ID): NodeState = {
      val localFeature = network.feature(id)
      val archData = network.archFeature(id)
      val neighbours = network.neighbourhood(id)
      val data = neighbours.map(id => NeighborhoodData(network.feature(id), network.state(id), archData(id))).toSeq
      gnn.eval(localFeature, data)
    }
  }
}
