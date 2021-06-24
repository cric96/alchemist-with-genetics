package it.unibo.gnn
import io.jenetics.engine.{Engine, EvolutionResult, EvolutionStatistics, Limits}
import io.jenetics.util.RandomRegistry
import io.jenetics.{DoubleGene, Genotype, Mutator, RouletteWheelSelector, SinglePointCrossover, TournamentSelector}
import it.unibo.gnn.evolutionary.{GNNCodec, JeneticsFacade}
import it.unibo.scafi.config.GridSettings
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.activations.Activation

import java.lang
object GnnSimulationTest extends App {
  val seed = 42
  val steady = 400
  val populationSize = 400
  RandomRegistry.random(new java.util.Random(seed))

  def spawnSimulation(program : AggregateProgram, length : Int = 100, network : Option[GraphNeuralNetwork] = None) : (NetworkSimulator, Map[ID, Double]) = {
    val simulator = simulatorFactory.gridLike(
      GridSettings(3, 3, 50, 50),
      60,
      seeds = Seeds(seed, seed, seed)
    )
    network.foreach(simulator.addSensor("network", _))
    val networkSimulator = simulator.asInstanceOf[NetworkSimulator] //unsafe
    simulator.addSensor("source", 0.0f)
    simulator.chgSensorValue("source", Set(1), 1.0f)
    simulator.addSensor("initialState", Array(-1.0f, -1.0f, -1.0f, -1.0f))

    (0 to length) foreach { _ => simulator.exec(program)}

    val results = simulator.exports().map { case (id, data) => id -> data.get.root[Double]() }
    (networkSimulator, results)
  }

  val hopCountProgram = new AggregateProgram with FieldUtils {
    override def main(): Any = rep(Double.PositiveInfinity) {
      value => mux(mid() == 0) { 0.0 } { minHoodPlus(nbr(value) + 1) }
    }
  }

  val (_, results) = spawnSimulation(hopCountProgram)
  val stateConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(7).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(4).build()
    ).build()

  val outputConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(5).nOut(8).build(),
      new DenseLayer.Builder().nIn(4).nOut(2).build(),
      new DenseLayer.Builder().nIn(2).nOut(1).build()
    ).build()

  val codec = GNNCodec(stateConfiguration, outputConfiguration)
  def fitness(genotype : Genotype[DoubleGene], codec : GNNCodec) : Double = {
    val gnn = codec.loadFromGenotype(genotype)
    val (_, gnnResults) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
    val fitness = gnnResults.map { case (id, v) => results(id) - v }.map { value => Math.abs(value) }.sum
    fitness
  }

  val factory = codec.genotypeFactory()
  val runner : Engine[DoubleGene, lang.Double] = JeneticsFacade.doubleEngine[DoubleGene](genotype => fitness(genotype, codec), factory)
    .populationSize(populationSize)
    .survivorsSelector(new TournamentSelector(5))
    .offspringSelector(new RouletteWheelSelector())
    .alterers(
      new Mutator(0.115),
      new SinglePointCrossover(0.16))
    .minimizing()
    .build()

  val statistics = EvolutionStatistics.ofNumber[lang.Double]()
  val result = runner
    .stream()
    .limit(Limits.bySteadyFitness[lang.Double](steady))
    .limit(Limits.byPopulationConvergence[lang.Double](0.1))
    .peek(e => statistics.accept(e))
    .peek(e => println(s"Generation ${e.generation()}, max fitness : ${e.bestFitness()}"))
    .collect(EvolutionResult.toBestPhenotype[DoubleGene, lang.Double])

  println(statistics)

  val gnn = codec.loadFromGenotype(result.genotype())
  val (_, gnnResult) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
  println(gnnResult)
  println(results)
}
