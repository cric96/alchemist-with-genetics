package it.unibo.gnn
import io.jenetics.engine.{Engine, EvolutionResult, EvolutionStatistics}
import io.jenetics.util.RandomRegistry
import io.jenetics.{DoubleGene, Genotype}
import it.unibo.gnn.evolutionary.{GNNCodec, JeneticsFacade}
import it.unibo.scafi.config.GridSettings
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.activations.Activation

import java.lang
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

  /*val stateConfiguration = new NeuralNetConfiguration.Builder()
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
    stateNetwork.init()
  outputNetwork.init()
  val gnn = GraphNeuralNetwork(stateNetwork, outputNetwork)
  networkSimulator.addSensor("network", gnn)
  val time = System.currentTimeMillis()
  (1 to 1000).foreach { _ => simulator.exec(new ScafiHopCountGNN()) }
  println(System.currentTimeMillis() - time)

  val stateNetwork = new MultiLayerNetwork(stateConfiguration)
  val outputNetwork = new MultiLayerNetwork(outputConfiguration)
  */
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

  val codec = GNNCodec(stateConfiguration, outputConfiguration)

  def fitness(genotype : Genotype[DoubleGene], codec : GNNCodec) : Double = {
    val gnn = codec.loadFromGenotype(genotype)
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
    networkSimulator.addSensor("network", gnn)
    (1 to 100) foreach { _ => networkSimulator.exec(new ScafiHopCountGNN) }
    Math.abs(networkSimulator.exports.apply(0).get.root[Double] - 0)
  }

  val factory = codec.genotypeFactory()
  val x : Comparable[Double] = 10.0
  val runner : Engine[DoubleGene, lang.Double] = JeneticsFacade.doubleEngine[DoubleGene](genotype => fitness(genotype, codec), factory)
    .populationSize(500)
    .build()
  val statistics = EvolutionStatistics.ofNumber[lang.Double]()
  val stream = runner
    .stream()
    .peek(e => statistics.accept(e))
    .limit(100)
    .collect(EvolutionResult.toBestPhenotype[DoubleGene, lang.Double])

  println(statistics)
}
