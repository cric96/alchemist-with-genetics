package it.unibo.gnn
import io.jenetics.util.RandomRegistry
import it.unibo.scafi.config.GridSettings
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
object GnnSimulationTest extends App {
  val seed = 42
  RandomRegistry.random(new java.util.Random(seed))
  val simulator = simulatorFactory.gridLike(
    GridSettings(4, 4),
    80,
    seeds = Seeds(seed, seed, seed)
  )
  val networkSimulator = simulator.asInstanceOf[NetworkSimulator] //unsafe
  networkSimulator.addSensor("source", 0.0f)
  networkSimulator.chgSensorValue("source", Set(1), 1.0f)
  networkSimulator.addSensor("initialState", Array(-1.0f, -1.0f, -1.0f, -1.0f))

  val stateConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(6).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(4).build()
    ).build()

  val outputConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(6).nOut(8).build(),
      new DenseLayer.Builder().nIn(4).nOut(2).build(),
      new DenseLayer.Builder().nIn(2).nOut(1).build()
    ).build()

  val stateNetwork = new MultiLayerNetwork(stateConfiguration)
  val outputNetwork = new MultiLayerNetwork(outputConfiguration)

  stateNetwork.init()
  outputNetwork.init()
  val gnn = GraphNeuralNetwork(stateNetwork, outputNetwork)
  networkSimulator.addSensor("network", gnn)
  (1 to 100).foreach { _ => simulator.exec(new ScafiHopCountGNN()) }

}
