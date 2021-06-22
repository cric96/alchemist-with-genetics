package it.unibo.gnn

import it.unibo.gnn.GraphNeuralNetwork.GraphLayer
import org.nd4j.linalg.activations.Activation
class GraphNeuralNetwork(layers : GraphLayer*)
object GraphNeuralNetwork {
  sealed class GraphLayer(val input : Int, val output : Int, val activation: Activation) {

  }

  type GraphLayerData = (Int, Activation)

  sealed case class GraphNetworkBuilder(val input : Int, private val layers : Seq[GraphLayerData] = Seq.empty) {
    def layer(output : Int, activation : Activation = Activation.RELU) = this.copy(layers = layers :+ (output, activation))
    //def build() : GraphNeuralNetwork = layers.
  }

}
