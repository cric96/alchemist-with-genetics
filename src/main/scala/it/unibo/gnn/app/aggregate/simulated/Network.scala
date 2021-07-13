package it.unibo.gnn.app.aggregate.simulated

import org.nd4j.linalg.api.ndarray.INDArray

trait Network[ID] {
  def feature(id: ID): INDArray
  def output(id: ID): INDArray
  def state(id : ID): INDArray
  def neighbourhood(id: ID): Set[ID]
  def archFeature(id: ID): Map[ID, INDArray]
}

object Network {
  type NetworkWithOps[ID] = Network[ID] with NetworkOps[ID]

  def apply[ID](
     nodes: Map[ID, INDArray],
     outputMap: Map[ID, INDArray],
     stateMap: Map[ID, INDArray],
     neighbourhoodMap: Map[ID, Set[ID]],
     archFeatureMap: Map[ID, Map[ID, INDArray]]
  ) : NetworkWithOps[ID] = {
    new NetworkImpl[ID](nodes, outputMap, stateMap, neighbourhoodMap, archFeatureMap)
  }
  trait NetworkOps[ID] {
    self : Network[ID] =>
    def updateFeature(id: ID, data: INDArray): NetworkWithOps[ID]
    def updateOutput(id: ID, data: INDArray): NetworkWithOps[ID]
    def updateNeighbour(center: ID, neighbours: Set[ID]): NetworkWithOps[ID]
    def updateArchFeature(link : (ID, ID), data : INDArray): NetworkWithOps[ID]
    def updateState(id: ID, state: INDArray): NetworkWithOps[ID]
  }

  private[Network] class NetworkImpl[ID](
      nodes: Map[ID, INDArray],
      outputMap: Map[ID, INDArray],
      stateMap: Map[ID, INDArray],
      neighbourhoodMap: Map[ID, Set[ID]],
      archFeatureMap: Map[ID, Map[ID, INDArray]]
    ) extends Network[ID] with NetworkOps[ID] {
    override def feature(id: ID): INDArray = nodes(id)

    override def output(id: ID): INDArray = outputMap(id)

    override def state(id: ID): INDArray = stateMap(id)

    override def neighbourhood(id: ID): Set[ID] = neighbourhoodMap(id)

    override def archFeature(id: ID): Map[ID, INDArray] = archFeatureMap(id)

    override def updateFeature(id: ID, data: INDArray): NetworkWithOps[ID] = new NetworkImpl(
      nodes + (id -> data),
      outputMap,
      stateMap,
      neighbourhoodMap,
      archFeatureMap
    )

    override def updateOutput(id: ID, data: INDArray): NetworkWithOps[ID] = new NetworkImpl(
      nodes,
      outputMap + (id -> data),
      stateMap,
      neighbourhoodMap,
      archFeatureMap
    )

    override def updateNeighbour(center: ID, neighbours: Set[ID]): NetworkWithOps[ID] = new NetworkImpl(
      nodes,
      outputMap,
      stateMap,
      neighbourhoodMap + (center -> neighbours),
      archFeatureMap
    )

    override def updateArchFeature(link: (ID, ID), data: INDArray): NetworkWithOps[ID] = new NetworkImpl(
      nodes,
      outputMap,
      stateMap,
      neighbourhoodMap,
      archFeatureMap // TODO
    )

    override def updateState(id: ID, state: INDArray): NetworkWithOps[ID] = new NetworkImpl(
      nodes,
      outputMap,
      stateMap + (id -> state),
      neighbourhoodMap,
      archFeatureMap
    )
  }
}
