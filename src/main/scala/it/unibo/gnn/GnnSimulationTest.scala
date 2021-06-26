package it.unibo.gnn
import io.jenetics._
import io.jenetics.engine.{Engine, EvolutionResult, EvolutionStatistics, Limits}
import io.jenetics.util.RandomRegistry
import io.jenetics.xml.Writers
import it.unibo.gnn.evolutionary.{GNNCodec, JeneticsFacade}
import it.unibo.scafi.config.GridSettings
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import java.io.FileOutputStream
import java.lang
import java.util.Random
import scala.jdk.CollectionConverters.IterableHasAsScala
import NetworkConfiguration._
object GnnSimulationTest extends App {
  // Constants
  /// environment constants
  val seed = 42
  val random = new Random(seed)
  val gridSetting = GridSettings(3, 3, 50, 50)
  /// scafi constants
  val radius = 60
  //// sensors
  val networkSensor = "network"
  val stateSensor = "initialState"
  val sourceSensor = "source"
  //// sensors values
  val initialSourceValue = 0.0f
  val initialStateValue = Array(-1.0f, -1.0f, -1.0f, 1.0f)
  val sourceOnValue = 1.0f
  val sourceId = 1
  // genetics constants
  val steady = 100
  val populationSize = 100
  RandomRegistry.random(random)
  // utility for creating ScaFi simulation, return the simulator and the exports produced
  def spawnSimulation(program : AggregateProgram, length : Int = 7, network : Option[GraphNeuralNetwork] = None) : (NetworkSimulator, Map[ID, Double]) = {
    val simulator = simulatorFactory.gridLike(gridSetting, radius, seeds = Seeds(seed, seed, seed))
    network.foreach(simulator.addSensor(networkSensor, _))
    val networkSimulator = simulator.asInstanceOf[NetworkSimulator] //unsafe
    simulator.addSensor(sourceSensor, initialSourceValue)
    simulator.chgSensorValue(sourceSensor, Set(sourceId), sourceOnValue)
    simulator.addSensor(stateSensor, initialStateValue)
    val ids = (0 to length).flatMap(_ => scala.util.Random.shuffle((0 until networkSimulator.ids.size).toList))
    ids foreach ( simulator.exec(program, program.main(), _))
    val results = simulator.exports().map { case (id, data) => id -> data.get.root[Double]() }
    (networkSimulator, results)
  }
  def statisticsByGeneration(e: EvolutionResult[DoubleGene, lang.Double]) : Unit = {
    val meanFitness = e.population().map(_.fitness).asScala.map[Double](a => a).sum / e.population().size()
    println(s"Generation ${e.generation()}, mean fitness : ${meanFitness}, best : ${e.bestFitness()}, worst : ${e.worstFitness()}")
  }
  // fitness used to valuate the solution
  def fitness(genotype : Genotype[DoubleGene], codec : GNNCodec) : Double = {
    val gnn = codec.loadFromGenotype(genotype)
    val (_, gnnResults) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
    val fitness = gnnResults.map { case (id, v) => references(id) - v }.map { value => Math.abs(value) }.sum
    fitness
  }
  // standard hop count program
  val hopCountProgram = new AggregateProgram with FieldUtils {
    override def main(): Any = rep(Double.PositiveInfinity) {
      value => mux(mid() == sourceId) { 0.0 } { minHoodPlus(nbr(value) + 1) }
    }
  }
  // create the ground truth
  val (_, references) = spawnSimulation(hopCountProgram)
  // genetics part
  /// factory of graph neural network encoded in terms of genotype
  val factory = codec.genotypeFactory()
  /// jenetics engine used, it describes the evolutionary algorithms that will be used.
  val runner : Engine[DoubleGene, lang.Double] = JeneticsFacade.doubleEngine[DoubleGene](genotype => fitness(genotype, codec), factory)
    .populationSize(populationSize)
    .survivorsSelector(new TournamentSelector(5))
    .offspringSelector(new RouletteWheelSelector())
    .alterers(
      new Mutator(0.315),
      new SinglePointCrossover(0.25))
    .minimizing()
    .build()
  //a mutable object used to gather evolutionary statistics
  val statistics = EvolutionStatistics.ofNumber[lang.Double]()
  //start the evolutionary algorithm
  val bestResult =
    RandomRegistry.`with`(random, _ => { //with is used to give the reproducibility of tests
      runner.stream()
        .limit(Limits.bySteadyFitness[lang.Double](steady))
        .peek(e => statistics.accept(e))
        .peek(statisticsByGeneration)
        .collect(EvolutionResult.toBestPhenotype[DoubleGene, lang.Double])
    })
  val gnn = codec.loadFromGenotype(bestResult.genotype())
  val (_, gnnResult) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
  bestResult.genotype()
  val file = new FileOutputStream("result.xml")
  Writers.Genotype.write[lang.Double, DoubleGene, DoubleChromosome](file, bestResult.genotype(), Writers.DoubleChromosome.writer())
  println(gnnResult)
  println(references)
  file.close()
}
