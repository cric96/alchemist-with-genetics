package it.unibo.gnn.app.program

import it.unibo.gnn.app.program.AggregateProgramSimulator.{Network, NetworkOps}
import it.unibo.gnn.model.GraphNeuralNetwork
import it.unibo.gnn.model.GraphNeuralNetwork.{NeighborhoodData, NodeState}
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * It simulates an aggregate program evaluation. It uses internally only the GNN interaction with the
 * neighbourhood.
 */
trait AggregateProgramSimulator {
  def evaluate(
    network: Network with NetworkOps,
    id: AggregateProgramSimulator.ID
  ): Network with NetworkOps
  def bulkEvaluation(
    network: Network with NetworkOps,
    ids: Set[AggregateProgramSimulator.ID]
  ): Network with NetworkOps
  def timeSeries(
    network: Network with NetworkOps,
    ids: Iterable[AggregateProgramSimulator.ID]
  ): Network with NetworkOps
  def timeSeriesAndBulk(
     network: Network with NetworkOps,
     evaluation: Iterable[Set[AggregateProgramSimulator.ID]]
  ): Network with NetworkOps
}

object AggregateProgramSimulator {
  type ID = Int
  def simulatorFromGNN(gnn: GraphNeuralNetwork) : AggregateProgramSimulator = new AggregateProgramSimulator {
    override def evaluate(
      network: Network with NetworkOps,
      id: ID
    ): Network with NetworkOps = {
      val result = graphNetworkEvaluation(network, id)
      network.updateState(id, result.state).updateOutput(id, result.output)
    }

    override def bulkEvaluation(
        network: Network with NetworkOps,
        ids: Set[ID]
      ): Network with NetworkOps = {
      val results = ids.map(id => id -> graphNetworkEvaluation(network, id))
      results.foldLeft(network){
        case(acc, (id, result)) => acc.updateState(id, result.state).updateOutput(id, result.output)
      }
    }

    override def timeSeries(
        network: Network with NetworkOps,
        ids: Iterable[ID]
      ): Network with NetworkOps = {
      ids.foldLeft(network){
        case (network, id) => evaluate(network, id)
      }
    }

    override def timeSeriesAndBulk(
      network: Network with NetworkOps,
      evaluations: Iterable[Set[ID]]
    ): Network with NetworkOps = {
      evaluations.foldLeft(network){
        case (network, ids) => bulkEvaluation(network, ids)
      }
    }

    def graphNetworkEvaluation(network: Network, id: ID): NodeState = {
      val localFeature = network.feature(id)
      val archData = network.archFeature(id)
      val neighbours = network.neighbourhood(id)
      val data = neighbours.map(id => NeighborhoodData(network.feature(id), network.state(id), archData(id))).toSeq
      gnn.eval(localFeature, data)
    }
  }
  trait Network {
    def feature(id: ID): INDArray
    def output(id: ID): INDArray
    def state(id : ID): INDArray
    def neighbourhood(id: ID): Set[ID]
    def archFeature(id: ID): Map[ID, INDArray]
  }

  trait NetworkOps {
    self : Network =>
    def updateFeature(id: ID, data: INDArray): Network with NetworkOps
    def updateOutput(id: ID, data: INDArray): Network with NetworkOps
    def updateNeighbour(center: ID, neighbours: Set[ID]): Network with NetworkOps
    def updateArchFeature(link : (ID, ID), data : INDArray): Network with NetworkOps
    def updateState(id: ID, state: INDArray): Network with NetworkOps
  }

  private[AggregateProgramSimulator] class NetworkImpl(
    nodes: Map[ID, INDArray],
    outputMap: Map[ID, INDArray],
    stateMap: Map[ID, INDArray],
    neighbourhoodMap: Map[ID, Set[ID]],
    archFeatureMap: Map[ID, Map[ID, INDArray]]
  ) extends Network with NetworkOps {
    override def feature(id: ID): INDArray = nodes(id)

    override def output(id: ID): INDArray = outputMap(id)

    override def state(id: ID): INDArray = stateMap(id)

    override def neighbourhood(id: ID): Set[ID] = neighbourhoodMap(id)

    override def archFeature(id: ID): Map[ID, INDArray] = archFeatureMap(id)

    override def updateFeature(id: ID, data: INDArray): Network = new NetworkImpl(
      nodes + (id -> data),
      outputMap,
      stateMap,
      neighbourhoodMap,
      archFeatureMap
    )

    override def updateOutput(id: ID, data: INDArray): Network = new NetworkImpl(
      nodes,
      outputMap + (id -> data),
      stateMap,
      neighbourhoodMap,
      archFeatureMap
    )

    override def updateNeighbour(center: ID, neighbours: Set[ID]): Network = new NetworkImpl(
      nodes,
      outputMap,
      stateMap,
      neighbourhoodMap + (center -> neighbours),
      archFeatureMap
    )

    override def updateArchFeature(link: (ID, ID), data: INDArray): Network = new NetworkImpl(
      nodes,
      outputMap,
      stateMap,
      neighbourhoodMap,
      archFeatureMap + (link -> data)
    )

    override def updateState(id: ID, state: INDArray): Network = new NetworkImpl(
      nodes,
      outputMap,
      stateMap + (id -> state),
      neighbourhoodMap,
      archFeatureMap
    )
  }
}
