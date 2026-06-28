//
// Created by keega on 9/16/2021.
// Extended with full RL bindings
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/search/Action.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/GameContext.h"
#include "game/Shop.h"
#include "game/Neow.h"
#include "combat/BattleContext.h"
#include "combat/Player.h"
#include "combat/Monster.h"
#include "combat/MonsterGroup.h"
#include "combat/CardInstance.h"
#include "combat/CardManager.h"
#include "combat/CardSelectInfo.h"
#include "combat/InputState.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/Potions.h"
#include "constants/MonsterIds.h"
#include "constants/Events.h"

#include "slaythespire.h"


using namespace sts;
namespace pybind = pybind11;

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "Slay the Spire simulator with full RL bindings";

    // ==================== Module-level functions ====================
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");

    // ==================== NNInterface ====================
    pybind::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    // ==================== Agent (ScumSearchAgent2) ====================
    pybind::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def("playout", &search::ScumSearchAgent2::playout);

    // ==================== Enums ====================

    // GameOutcome
    pybind::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    // Battle Outcome
    pybind::enum_<Outcome>(m, "BattleOutcome")
        .value("UNDECIDED", Outcome::UNDECIDED)
        .value("PLAYER_VICTORY", Outcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", Outcome::PLAYER_LOSS);

    // ScreenState
    pybind::enum_<ScreenState> screenState(m, "ScreenState");
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    // CharacterClass
    pybind::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    // Room
    pybind::enum_<Room> roomEnum(m, "Room");
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    // CardRarity
    pybind::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    // CardColor
    pybind::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    // CardType
    pybind::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    // InputState (battle)
    pybind::enum_<InputState>(m, "InputState")
        .value("EXECUTING_ACTIONS", InputState::EXECUTING_ACTIONS)
        .value("PLAYER_NORMAL", InputState::PLAYER_NORMAL)
        .value("CARD_SELECT", InputState::CARD_SELECT);

    // Stance
    pybind::enum_<Stance>(m, "Stance")
        .value("NEUTRAL", Stance::NEUTRAL)
        .value("CALM", Stance::CALM)
        .value("WRATH", Stance::WRATH)
        .value("DIVINITY", Stance::DIVINITY);

    // CardSelectScreenType
    pybind::enum_<CardSelectScreenType>(m, "CardSelectScreenType")
        .value("INVALID", CardSelectScreenType::INVALID)
        .value("TRANSFORM", CardSelectScreenType::TRANSFORM)
        .value("TRANSFORM_UPGRADE", CardSelectScreenType::TRANSFORM_UPGRADE)
        .value("UPGRADE", CardSelectScreenType::UPGRADE)
        .value("REMOVE", CardSelectScreenType::REMOVE)
        .value("DUPLICATE", CardSelectScreenType::DUPLICATE)
        .value("OBTAIN", CardSelectScreenType::OBTAIN)
        .value("BOTTLE", CardSelectScreenType::BOTTLE)
        .value("BONFIRE_SPIRITS", CardSelectScreenType::BONFIRE_SPIRITS);

    // CardSelectTask (battle card selection)
    pybind::enum_<CardSelectTask>(m, "CardSelectTask")
        .value("INVALID", CardSelectTask::INVALID)
        .value("ARMAMENTS", CardSelectTask::ARMAMENTS)
        .value("CODEX", CardSelectTask::CODEX)
        .value("DISCOVERY", CardSelectTask::DISCOVERY)
        .value("DUAL_WIELD", CardSelectTask::DUAL_WIELD)
        .value("EXHAUST_ONE", CardSelectTask::EXHAUST_ONE)
        .value("EXHAUST_MANY", CardSelectTask::EXHAUST_MANY)
        .value("EXHUME", CardSelectTask::EXHUME)
        .value("FORETHOUGHT", CardSelectTask::FORETHOUGHT)
        .value("GAMBLE", CardSelectTask::GAMBLE)
        .value("HEADBUTT", CardSelectTask::HEADBUTT)
        .value("HOLOGRAM", CardSelectTask::HOLOGRAM)
        .value("LIQUID_MEMORIES_POTION", CardSelectTask::LIQUID_MEMORIES_POTION)
        .value("MEDITATE", CardSelectTask::MEDITATE)
        .value("NIGHTMARE", CardSelectTask::NIGHTMARE)
        .value("RECYCLE", CardSelectTask::RECYCLE)
        .value("SECRET_TECHNIQUE", CardSelectTask::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardSelectTask::SECRET_WEAPON)
        .value("SEEK", CardSelectTask::SEEK)
        .value("SETUP", CardSelectTask::SETUP)
        .value("WARCRY", CardSelectTask::WARCRY);

    // ActionType (for search::Action)
    pybind::enum_<search::ActionType>(m, "ActionType")
        .value("CARD", search::ActionType::CARD)
        .value("POTION", search::ActionType::POTION)
        .value("SINGLE_CARD_SELECT", search::ActionType::SINGLE_CARD_SELECT)
        .value("MULTI_CARD_SELECT", search::ActionType::MULTI_CARD_SELECT)
        .value("END_TURN", search::ActionType::END_TURN);

    // Neow::Bonus
    pybind::enum_<Neow::Bonus>(m, "NeowBonus")
        .value("THREE_CARDS", Neow::Bonus::THREE_CARDS)
        .value("ONE_RANDOM_RARE_CARD", Neow::Bonus::ONE_RANDOM_RARE_CARD)
        .value("REMOVE_CARD", Neow::Bonus::REMOVE_CARD)
        .value("UPGRADE_CARD", Neow::Bonus::UPGRADE_CARD)
        .value("TRANSFORM_CARD", Neow::Bonus::TRANSFORM_CARD)
        .value("RANDOM_COLORLESS", Neow::Bonus::RANDOM_COLORLESS)
        .value("THREE_SMALL_POTIONS", Neow::Bonus::THREE_SMALL_POTIONS)
        .value("RANDOM_COMMON_RELIC", Neow::Bonus::RANDOM_COMMON_RELIC)
        .value("TEN_PERCENT_HP_BONUS", Neow::Bonus::TEN_PERCENT_HP_BONUS)
        .value("THREE_ENEMY_KILL", Neow::Bonus::THREE_ENEMY_KILL)
        .value("HUNDRED_GOLD", Neow::Bonus::HUNDRED_GOLD)
        .value("RANDOM_COLORLESS_2", Neow::Bonus::RANDOM_COLORLESS_2)
        .value("REMOVE_TWO", Neow::Bonus::REMOVE_TWO)
        .value("ONE_RARE_RELIC", Neow::Bonus::ONE_RARE_RELIC)
        .value("THREE_RARE_CARDS", Neow::Bonus::THREE_RARE_CARDS)
        .value("TWO_FIFTY_GOLD", Neow::Bonus::TWO_FIFTY_GOLD)
        .value("TRANSFORM_TWO_CARDS", Neow::Bonus::TRANSFORM_TWO_CARDS)
        .value("TWENTY_PERCENT_HP_BONUS", Neow::Bonus::TWENTY_PERCENT_HP_BONUS)
        .value("BOSS_RELIC", Neow::Bonus::BOSS_RELIC)
        .value("INVALID", Neow::Bonus::INVALID);

    // Neow::Drawback
    pybind::enum_<Neow::Drawback>(m, "NeowDrawback")
        .value("INVALID", Neow::Drawback::INVALID)
        .value("NONE", Neow::Drawback::NONE)
        .value("TEN_PERCENT_HP_LOSS", Neow::Drawback::TEN_PERCENT_HP_LOSS)
        .value("NO_GOLD", Neow::Drawback::NO_GOLD)
        .value("CURSE", Neow::Drawback::CURSE)
        .value("PERCENT_DAMAGE", Neow::Drawback::PERCENT_DAMAGE)
        .value("LOSE_STARTER_RELIC", Neow::Drawback::LOSE_STARTER_RELIC);

    // PlayerStatus
    pybind::enum_<PlayerStatus>(m, "PlayerStatus")
        .value("INVALID", PlayerStatus::INVALID)
        .value("VULNERABLE", PlayerStatus::VULNERABLE)
        .value("WEAK", PlayerStatus::WEAK)
        .value("FRAIL", PlayerStatus::FRAIL)
        .value("STRENGTH", PlayerStatus::STRENGTH)
        .value("DEXTERITY", PlayerStatus::DEXTERITY)
        .value("ARTIFACT", PlayerStatus::ARTIFACT)
        .value("FOCUS", PlayerStatus::FOCUS)
        .value("INTANGIBLE", PlayerStatus::INTANGIBLE)
        .value("BARRICADE", PlayerStatus::BARRICADE)
        .value("CORRUPTION", PlayerStatus::CORRUPTION)
        .value("ENTANGLED", PlayerStatus::ENTANGLED)
        .value("NO_DRAW", PlayerStatus::NO_DRAW)
        .value("CONFUSED", PlayerStatus::CONFUSED)
        .value("DRAW_REDUCTION", PlayerStatus::DRAW_REDUCTION)
        .value("METALLICIZE", PlayerStatus::METALLICIZE)
        .value("PLATED_ARMOR", PlayerStatus::PLATED_ARMOR)
        .value("THORNS", PlayerStatus::THORNS)
        .value("REGEN", PlayerStatus::REGEN)
        .value("RITUAL", PlayerStatus::RITUAL)
        .value("PEN_NIB", PlayerStatus::PEN_NIB)
        .value("VIGOR", PlayerStatus::VIGOR)
        .value("MANTRA", PlayerStatus::MANTRA)
        .value("BUFFER", PlayerStatus::BUFFER)
        .value("DOUBLE_DAMAGE", PlayerStatus::DOUBLE_DAMAGE)
        .value("DOUBLE_TAP", PlayerStatus::DOUBLE_TAP)
        .value("BURST", PlayerStatus::BURST)
        .value("BLUR", PlayerStatus::BLUR)
        .value("ECHO_FORM", PlayerStatus::ECHO_FORM)
        .value("DEMON_FORM", PlayerStatus::DEMON_FORM)
        .value("FLAME_BARRIER", PlayerStatus::FLAME_BARRIER)
        .value("RAGE", PlayerStatus::RAGE)
        .value("COMBUST", PlayerStatus::COMBUST)
        .value("DARK_EMBRACE", PlayerStatus::DARK_EMBRACE)
        .value("EVOLVE", PlayerStatus::EVOLVE)
        .value("FEEL_NO_PAIN", PlayerStatus::FEEL_NO_PAIN)
        .value("FIRE_BREATHING", PlayerStatus::FIRE_BREATHING)
        .value("RUPTURE", PlayerStatus::RUPTURE)
        .value("JUGGERNAUT", PlayerStatus::JUGGERNAUT)
        .value("BRUTALITY", PlayerStatus::BRUTALITY)
        .value("THE_BOMB", PlayerStatus::THE_BOMB)
        .value("NOXIOUS_FUMES", PlayerStatus::NOXIOUS_FUMES)
        .value("ENVENOM", PlayerStatus::ENVENOM)
        .value("ACCURACY", PlayerStatus::ACCURACY)
        .value("AFTER_IMAGE", PlayerStatus::AFTER_IMAGE)
        .value("INFINITE_BLADES", PlayerStatus::INFINITE_BLADES)
        .value("THOUSAND_CUTS", PlayerStatus::THOUSAND_CUTS)
        .value("TOOLS_OF_THE_TRADE", PlayerStatus::TOOLS_OF_THE_TRADE)
        .value("PHANTASMAL", PlayerStatus::PHANTASMAL)
        .value("WRAITH_FORM", PlayerStatus::WRAITH_FORM)
        .value("STATIC_DISCHARGE", PlayerStatus::STATIC_DISCHARGE)
        .value("LOOP", PlayerStatus::LOOP)
        .value("CREATIVE_AI", PlayerStatus::CREATIVE_AI)
        .value("HELLO_WORLD", PlayerStatus::HELLO_WORLD)
        .value("ELECTRO", PlayerStatus::ELECTRO)
        .value("BIAS", PlayerStatus::BIAS)
        .value("ESTABLISHMENT", PlayerStatus::ESTABLISHMENT)
        .value("FORESIGHT", PlayerStatus::FORESIGHT)
        .value("LIKE_WATER", PlayerStatus::LIKE_WATER)
        .value("DEVOTION", PlayerStatus::DEVOTION)
        .value("DEVA", PlayerStatus::DEVA)
        .value("MASTER_REALITY", PlayerStatus::MASTER_REALITY)
        .value("BATTLE_HYMN", PlayerStatus::BATTLE_HYMN)
        .value("OMEGA", PlayerStatus::OMEGA)
        .value("WAVE_OF_THE_HAND", PlayerStatus::WAVE_OF_THE_HAND)
        .value("FASTING", PlayerStatus::FASTING)
        .value("ENERGIZED", PlayerStatus::ENERGIZED)
        .value("DRAW_CARD_NEXT_TURN", PlayerStatus::DRAW_CARD_NEXT_TURN)
        .value("NEXT_TURN_BLOCK", PlayerStatus::NEXT_TURN_BLOCK)
        .value("SURROUNDED", PlayerStatus::SURROUNDED)
        .value("HEX", PlayerStatus::HEX)
        .value("CONSTRICTED", PlayerStatus::CONSTRICTED);

    // MonsterStatus
    pybind::enum_<MonsterStatus>(m, "MonsterStatus")
        .value("ARTIFACT", MonsterStatus::ARTIFACT)
        .value("BLOCK_RETURN", MonsterStatus::BLOCK_RETURN)
        .value("CHOKED", MonsterStatus::CHOKED)
        .value("CORPSE_EXPLOSION", MonsterStatus::CORPSE_EXPLOSION)
        .value("LOCK_ON", MonsterStatus::LOCK_ON)
        .value("MARK", MonsterStatus::MARK)
        .value("METALLICIZE", MonsterStatus::METALLICIZE)
        .value("PLATED_ARMOR", MonsterStatus::PLATED_ARMOR)
        .value("POISON", MonsterStatus::POISON)
        .value("REGEN", MonsterStatus::REGEN)
        .value("SHACKLED", MonsterStatus::SHACKLED)
        .value("STRENGTH", MonsterStatus::STRENGTH)
        .value("VULNERABLE", MonsterStatus::VULNERABLE)
        .value("WEAK", MonsterStatus::WEAK)
        .value("ANGRY", MonsterStatus::ANGRY)
        .value("BEAT_OF_DEATH", MonsterStatus::BEAT_OF_DEATH)
        .value("CURIOSITY", MonsterStatus::CURIOSITY)
        .value("CURL_UP", MonsterStatus::CURL_UP)
        .value("ENRAGE", MonsterStatus::ENRAGE)
        .value("FADING", MonsterStatus::FADING)
        .value("FLIGHT", MonsterStatus::FLIGHT)
        .value("GENERIC_STRENGTH_UP", MonsterStatus::GENERIC_STRENGTH_UP)
        .value("INTANGIBLE", MonsterStatus::INTANGIBLE)
        .value("MALLEABLE", MonsterStatus::MALLEABLE)
        .value("MODE_SHIFT", MonsterStatus::MODE_SHIFT)
        .value("RITUAL", MonsterStatus::RITUAL)
        .value("SLOW", MonsterStatus::SLOW)
        .value("SPORE_CLOUD", MonsterStatus::SPORE_CLOUD)
        .value("THIEVERY", MonsterStatus::THIEVERY)
        .value("THORNS", MonsterStatus::THORNS)
        .value("TIME_WARP", MonsterStatus::TIME_WARP)
        .value("INVINCIBLE", MonsterStatus::INVINCIBLE)
        .value("REACTIVE", MonsterStatus::REACTIVE)
        .value("SHARP_HIDE", MonsterStatus::SHARP_HIDE)
        .value("ASLEEP", MonsterStatus::ASLEEP)
        .value("BARRICADE", MonsterStatus::BARRICADE)
        .value("MINION", MonsterStatus::MINION)
        .value("MINION_LEADER", MonsterStatus::MINION_LEADER)
        .value("PAINFUL_STABS", MonsterStatus::PAINFUL_STABS)
        .value("REGROW", MonsterStatus::REGROW)
        .value("SHIFTING", MonsterStatus::SHIFTING)
        .value("STASIS", MonsterStatus::STASIS)
        .value("INVALID", MonsterStatus::INVALID);

    // Potion
    pybind::enum_<Potion>(m, "Potion")
        .value("INVALID", Potion::INVALID)
        .value("EMPTY_POTION_SLOT", Potion::EMPTY_POTION_SLOT)
        .value("AMBROSIA", Potion::AMBROSIA)
        .value("ANCIENT_POTION", Potion::ANCIENT_POTION)
        .value("ATTACK_POTION", Potion::ATTACK_POTION)
        .value("BLESSING_OF_THE_FORGE", Potion::BLESSING_OF_THE_FORGE)
        .value("BLOCK_POTION", Potion::BLOCK_POTION)
        .value("BLOOD_POTION", Potion::BLOOD_POTION)
        .value("BOTTLED_MIRACLE", Potion::BOTTLED_MIRACLE)
        .value("COLORLESS_POTION", Potion::COLORLESS_POTION)
        .value("CULTIST_POTION", Potion::CULTIST_POTION)
        .value("CUNNING_POTION", Potion::CUNNING_POTION)
        .value("DEXTERITY_POTION", Potion::DEXTERITY_POTION)
        .value("DISTILLED_CHAOS", Potion::DISTILLED_CHAOS)
        .value("DUPLICATION_POTION", Potion::DUPLICATION_POTION)
        .value("ELIXIR_POTION", Potion::ELIXIR_POTION)
        .value("ENERGY_POTION", Potion::ENERGY_POTION)
        .value("ENTROPIC_BREW", Potion::ENTROPIC_BREW)
        .value("ESSENCE_OF_DARKNESS", Potion::ESSENCE_OF_DARKNESS)
        .value("ESSENCE_OF_STEEL", Potion::ESSENCE_OF_STEEL)
        .value("EXPLOSIVE_POTION", Potion::EXPLOSIVE_POTION)
        .value("FAIRY_POTION", Potion::FAIRY_POTION)
        .value("FEAR_POTION", Potion::FEAR_POTION)
        .value("FIRE_POTION", Potion::FIRE_POTION)
        .value("FLEX_POTION", Potion::FLEX_POTION)
        .value("FOCUS_POTION", Potion::FOCUS_POTION)
        .value("FRUIT_JUICE", Potion::FRUIT_JUICE)
        .value("GAMBLERS_BREW", Potion::GAMBLERS_BREW)
        .value("GHOST_IN_A_JAR", Potion::GHOST_IN_A_JAR)
        .value("HEART_OF_IRON", Potion::HEART_OF_IRON)
        .value("LIQUID_BRONZE", Potion::LIQUID_BRONZE)
        .value("LIQUID_MEMORIES", Potion::LIQUID_MEMORIES)
        .value("POISON_POTION", Potion::POISON_POTION)
        .value("POTION_OF_CAPACITY", Potion::POTION_OF_CAPACITY)
        .value("POWER_POTION", Potion::POWER_POTION)
        .value("REGEN_POTION", Potion::REGEN_POTION)
        .value("SKILL_POTION", Potion::SKILL_POTION)
        .value("SMOKE_BOMB", Potion::SMOKE_BOMB)
        .value("SNECKO_OIL", Potion::SNECKO_OIL)
        .value("SPEED_POTION", Potion::SPEED_POTION)
        .value("STANCE_POTION", Potion::STANCE_POTION)
        .value("STRENGTH_POTION", Potion::STRENGTH_POTION)
        .value("SWIFT_POTION", Potion::SWIFT_POTION)
        .value("WEAK_POTION", Potion::WEAK_POTION);

    // MonsterId
    pybind::enum_<MonsterId>(m, "MonsterId")
        .value("INVALID", MonsterId::INVALID)
        .value("ACID_SLIME_L", MonsterId::ACID_SLIME_L)
        .value("ACID_SLIME_M", MonsterId::ACID_SLIME_M)
        .value("ACID_SLIME_S", MonsterId::ACID_SLIME_S)
        .value("AWAKENED_ONE", MonsterId::AWAKENED_ONE)
        .value("BEAR", MonsterId::BEAR)
        .value("BLUE_SLAVER", MonsterId::BLUE_SLAVER)
        .value("BOOK_OF_STABBING", MonsterId::BOOK_OF_STABBING)
        .value("BRONZE_AUTOMATON", MonsterId::BRONZE_AUTOMATON)
        .value("BRONZE_ORB", MonsterId::BRONZE_ORB)
        .value("BYRD", MonsterId::BYRD)
        .value("CENTURION", MonsterId::CENTURION)
        .value("CHOSEN", MonsterId::CHOSEN)
        .value("CORRUPT_HEART", MonsterId::CORRUPT_HEART)
        .value("CULTIST", MonsterId::CULTIST)
        .value("DAGGER", MonsterId::DAGGER)
        .value("DARKLING", MonsterId::DARKLING)
        .value("DECA", MonsterId::DECA)
        .value("DONU", MonsterId::DONU)
        .value("EXPLODER", MonsterId::EXPLODER)
        .value("FAT_GREMLIN", MonsterId::FAT_GREMLIN)
        .value("FUNGI_BEAST", MonsterId::FUNGI_BEAST)
        .value("GIANT_HEAD", MonsterId::GIANT_HEAD)
        .value("GREEN_LOUSE", MonsterId::GREEN_LOUSE)
        .value("GREMLIN_LEADER", MonsterId::GREMLIN_LEADER)
        .value("GREMLIN_NOB", MonsterId::GREMLIN_NOB)
        .value("GREMLIN_WIZARD", MonsterId::GREMLIN_WIZARD)
        .value("HEXAGHOST", MonsterId::HEXAGHOST)
        .value("JAW_WORM", MonsterId::JAW_WORM)
        .value("LAGAVULIN", MonsterId::LAGAVULIN)
        .value("LOOTER", MonsterId::LOOTER)
        .value("MAD_GREMLIN", MonsterId::MAD_GREMLIN)
        .value("MUGGER", MonsterId::MUGGER)
        .value("MYSTIC", MonsterId::MYSTIC)
        .value("NEMESIS", MonsterId::NEMESIS)
        .value("ORB_WALKER", MonsterId::ORB_WALKER)
        .value("POINTY", MonsterId::POINTY)
        .value("RED_LOUSE", MonsterId::RED_LOUSE)
        .value("RED_SLAVER", MonsterId::RED_SLAVER)
        .value("REPTOMANCER", MonsterId::REPTOMANCER)
        .value("REPULSOR", MonsterId::REPULSOR)
        .value("ROMEO", MonsterId::ROMEO)
        .value("SENTRY", MonsterId::SENTRY)
        .value("SHELLED_PARASITE", MonsterId::SHELLED_PARASITE)
        .value("SHIELD_GREMLIN", MonsterId::SHIELD_GREMLIN)
        .value("SLIME_BOSS", MonsterId::SLIME_BOSS)
        .value("SNAKE_PLANT", MonsterId::SNAKE_PLANT)
        .value("SNEAKY_GREMLIN", MonsterId::SNEAKY_GREMLIN)
        .value("SNECKO", MonsterId::SNECKO)
        .value("SPHERIC_GUARDIAN", MonsterId::SPHERIC_GUARDIAN)
        .value("SPIKER", MonsterId::SPIKER)
        .value("SPIKE_SLIME_L", MonsterId::SPIKE_SLIME_L)
        .value("SPIKE_SLIME_M", MonsterId::SPIKE_SLIME_M)
        .value("SPIKE_SLIME_S", MonsterId::SPIKE_SLIME_S)
        .value("SPIRE_GROWTH", MonsterId::SPIRE_GROWTH)
        .value("SPIRE_SHIELD", MonsterId::SPIRE_SHIELD)
        .value("SPIRE_SPEAR", MonsterId::SPIRE_SPEAR)
        .value("TASKMASTER", MonsterId::TASKMASTER)
        .value("THE_CHAMP", MonsterId::THE_CHAMP)
        .value("THE_COLLECTOR", MonsterId::THE_COLLECTOR)
        .value("THE_GUARDIAN", MonsterId::THE_GUARDIAN)
        .value("THE_MAW", MonsterId::THE_MAW)
        .value("TIME_EATER", MonsterId::TIME_EATER)
        .value("TORCH_HEAD", MonsterId::TORCH_HEAD)
        .value("TRANSIENT", MonsterId::TRANSIENT)
        .value("WRITHING_MASS", MonsterId::WRITHING_MASS);

    // Event
    pybind::enum_<Event>(m, "Event")
        .value("INVALID", Event::INVALID)
        .value("MONSTER", Event::MONSTER)
        .value("REST", Event::REST)
        .value("SHOP", Event::SHOP)
        .value("TREASURE", Event::TREASURE)
        .value("NEOW", Event::NEOW)
        .value("OMINOUS_FORGE", Event::OMINOUS_FORGE)
        .value("PLEADING_VAGRANT", Event::PLEADING_VAGRANT)
        .value("ANCIENT_WRITING", Event::ANCIENT_WRITING)
        .value("OLD_BEGGAR", Event::OLD_BEGGAR)
        .value("BIG_FISH", Event::BIG_FISH)
        .value("BONFIRE_SPIRITS", Event::BONFIRE_SPIRITS)
        .value("COLOSSEUM", Event::COLOSSEUM)
        .value("CURSED_TOME", Event::CURSED_TOME)
        .value("DEAD_ADVENTURER", Event::DEAD_ADVENTURER)
        .value("DESIGNER_IN_SPIRE", Event::DESIGNER_IN_SPIRE)
        .value("AUGMENTER", Event::AUGMENTER)
        .value("DUPLICATOR", Event::DUPLICATOR)
        .value("FACE_TRADER", Event::FACE_TRADER)
        .value("FALLING", Event::FALLING)
        .value("FORGOTTEN_ALTAR", Event::FORGOTTEN_ALTAR)
        .value("THE_DIVINE_FOUNTAIN", Event::THE_DIVINE_FOUNTAIN)
        .value("GHOSTS", Event::GHOSTS)
        .value("GOLDEN_IDOL", Event::GOLDEN_IDOL)
        .value("GOLDEN_SHRINE", Event::GOLDEN_SHRINE)
        .value("WING_STATUE", Event::WING_STATUE)
        .value("KNOWING_SKULL", Event::KNOWING_SKULL)
        .value("LAB", Event::LAB)
        .value("THE_SSSSSERPENT", Event::THE_SSSSSERPENT)
        .value("LIVING_WALL", Event::LIVING_WALL)
        .value("MASKED_BANDITS", Event::MASKED_BANDITS)
        .value("MATCH_AND_KEEP", Event::MATCH_AND_KEEP)
        .value("MINDBLOOM", Event::MINDBLOOM)
        .value("HYPNOTIZING_COLORED_MUSHROOMS", Event::HYPNOTIZING_COLORED_MUSHROOMS)
        .value("MYSTERIOUS_SPHERE", Event::MYSTERIOUS_SPHERE)
        .value("THE_NEST", Event::THE_NEST)
        .value("NLOTH", Event::NLOTH)
        .value("NOTE_FOR_YOURSELF", Event::NOTE_FOR_YOURSELF)
        .value("PURIFIER", Event::PURIFIER)
        .value("SCRAP_OOZE", Event::SCRAP_OOZE)
        .value("SECRET_PORTAL", Event::SECRET_PORTAL)
        .value("SENSORY_STONE", Event::SENSORY_STONE)
        .value("SHINING_LIGHT", Event::SHINING_LIGHT)
        .value("THE_CLERIC", Event::THE_CLERIC)
        .value("THE_JOUST", Event::THE_JOUST)
        .value("THE_LIBRARY", Event::THE_LIBRARY)
        .value("THE_MAUSOLEUM", Event::THE_MAUSOLEUM)
        .value("THE_MOAI_HEAD", Event::THE_MOAI_HEAD)
        .value("THE_WOMAN_IN_BLUE", Event::THE_WOMAN_IN_BLUE)
        .value("TOMB_OF_LORD_RED_MASK", Event::TOMB_OF_LORD_RED_MASK)
        .value("TRANSMORGRIFIER", Event::TRANSMORGRIFIER)
        .value("UPGRADE_SHRINE", Event::UPGRADE_SHRINE)
        .value("VAMPIRES", Event::VAMPIRES)
        .value("WE_MEET_AGAIN", Event::WE_MEET_AGAIN)
        .value("WHEEL_OF_CHANGE", Event::WHEEL_OF_CHANGE)
        .value("WINDING_HALLS", Event::WINDING_HALLS)
        .value("WORLD_OF_GOOP", Event::WORLD_OF_GOOP);

    // CardId - Only Ironclad (RED), Colorless, Curses, and Status cards
    // Silent (GREEN), Defect (BLUE), and Watcher (PURPLE) cards are not supported
    pybind::enum_<CardId>(m, "CardId")
        .value("INVALID", CardId::INVALID)
        // === IRONCLAD (RED) CARDS ===
        .value("ANGER", CardId::ANGER)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BERSERK", CardId::BERSERK)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CLASH", CardId::CLASH)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("COMBUST", CardId::COMBUST)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DISARM", CardId::DISARM)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLEX", CardId::FLEX)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFLAME", CardId::INFLAME)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("OFFERING", CardId::OFFERING)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PUMMEL", CardId::PUMMEL)
        .value("RAGE", CardId::RAGE)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REAPER", CardId::REAPER)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RUPTURE", CardId::RUPTURE)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("WARCRY", CardId::WARCRY)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        // === COLORLESS CARDS ===
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BETA", CardId::BETA)
        .value("BITE", CardId::BITE)
        .value("BLIND", CardId::BLIND)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FINESSE", CardId::FINESSE)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JAX", CardId::JAX)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PURITY", CardId::PURITY)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SHIV", CardId::SHIV)
        .value("SMITE", CardId::SMITE)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("VIOLENCE", CardId::VIOLENCE)
        // === CURSES ===
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("DECAY", CardId::DECAY)
        .value("DOUBT", CardId::DOUBT)
        .value("INJURY", CardId::INJURY)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NORMALITY", CardId::NORMALITY)
        .value("PAIN", CardId::PAIN)
        .value("PARASITE", CardId::PARASITE)
        .value("PRIDE", CardId::PRIDE)
        .value("REGRET", CardId::REGRET)
        .value("SHAME", CardId::SHAME)
        .value("WRITHE", CardId::WRITHE)
        // === STATUS CARDS ===
        .value("BURN", CardId::BURN)
        .value("DAZED", CardId::DAZED)
        .value("SLIMED", CardId::SLIMED)
        .value("VOID", CardId::VOID)
        .value("WOUND", CardId::WOUND);

    // MonsterEncounter
    pybind::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter");
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    // RelicId
    pybind::enum_<RelicId> relicEnum(m, "RelicId");
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    // ==================== Card (deck card) ====================
    pybind::class_<Card> card(m, "Card");
    card.def(pybind::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    // ==================== CardInstance (battle card) ====================
    pybind::class_<CardInstance>(m, "CardInstance")
        .def(pybind::init<CardId, bool>(), pybind::arg("id"), pybind::arg("upgraded") = false)
        .def(pybind::init<const Card&>())
        .def("__repr__", [](const CardInstance &c) {
            std::ostringstream oss;
            c.printSimpleDesc(oss);
            return "<CardInstance " + oss.str() + ">";
        })
        .def_property_readonly("id", &CardInstance::getId)
        .def_property_readonly("type", &CardInstance::getType)
        .def_property_readonly("name", &CardInstance::getName)
        .def_property_readonly("unique_id", &CardInstance::getUniqueId)
        .def_property_readonly("upgraded", &CardInstance::isUpgraded)
        .def_property_readonly("upgrade_count", &CardInstance::getUpgradeCount)
        .def_property_readonly("can_upgrade", &CardInstance::canUpgrade)
        .def_property_readonly("ethereal", &CardInstance::isEthereal)
        .def_property_readonly("is_strike_card", &CardInstance::isStrikeCard)
        .def_property_readonly("exhausts", &CardInstance::doesExhaust)
        .def_property_readonly("has_self_retain", &CardInstance::hasSelfRetain)
        .def_property_readonly("requires_target", &CardInstance::requiresTarget)
        .def_property_readonly("is_x_cost", &CardInstance::isXCost)
        .def_readwrite("cost", &CardInstance::cost)
        .def_readwrite("cost_for_turn", &CardInstance::costForTurn)
        .def_readwrite("free_to_play_once", &CardInstance::freeToPlayOnce)
        .def_readwrite("retain", &CardInstance::retain)
        // Per-instance accumulator used by Searing Blow (upgrade level), Rampage
        // (bonus damage), Ritual Dagger (damage), Genetic Algorithm (block); 0 otherwise.
        .def_readonly("special_data", &CardInstance::specialData)
        .def("uses_special_data", &CardInstance::usesSpecialData)
        .def("is_free_to_play", &CardInstance::isFreeToPlay)
        .def("can_use_on_any_target", &CardInstance::canUseOnAnyTarget)
        .def("can_use", &CardInstance::canUse, pybind::arg("bc"), pybind::arg("target"), pybind::arg("in_autoplay") = false);

    // ==================== RelicInstance ====================
    pybind::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    // ==================== Neow::Option ====================
    pybind::class_<Neow::Option>(m, "NeowOption")
        .def_readonly("bonus", &Neow::Option::r)
        .def_readonly("drawback", &Neow::Option::d)
        .def("__repr__", [](const Neow::Option &opt) {
            return "<NeowOption bonus=" + std::to_string(static_cast<int>(opt.r)) +
                   " drawback=" + std::to_string(static_cast<int>(opt.d)) + ">";
        });

    // ==================== Shop ====================
    pybind::class_<Shop>(m, "Shop")
        .def_property_readonly("cards", [](const Shop &s) {
            return std::vector<Card>(std::begin(s.cards), std::end(s.cards));
        })
        .def_property_readonly("potions", [](const Shop &s) {
            return std::vector<Potion>(std::begin(s.potions), std::end(s.potions));
        })
        .def_property_readonly("relics", [](const Shop &s) {
            return std::vector<RelicId>(std::begin(s.relics), std::end(s.relics));
        })
        .def_property_readonly("prices", [](const Shop &s) {
            return std::vector<int>(std::begin(s.prices), std::end(s.prices));
        })
        .def_readonly("remove_cost", &Shop::removeCost)
        .def("card_price", pybind::overload_cast<int>(&Shop::cardPrice, pybind::const_))
        .def("relic_price", pybind::overload_cast<int>(&Shop::relicPrice, pybind::const_))
        .def("potion_price", pybind::overload_cast<int>(&Shop::potionPrice, pybind::const_))
        .def("buy_card", &Shop::buyCard)
        .def("buy_relic", &Shop::buyRelic)
        .def("buy_potion", &Shop::buyPotion)
        .def("buy_card_remove", &Shop::buyCardRemove);

    // ==================== SpireMap ====================
    pybind::class_<Map> map(m, "SpireMap");
    map.def(pybind::init<std::uint64_t, int, int, bool>());
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    // ==================== Player (battle state) ====================
    pybind::class_<Player>(m, "Player")
        .def_readonly("character_class", &Player::cc)
        .def_readonly("cur_hp", &Player::curHp)
        .def_readonly("max_hp", &Player::maxHp)
        .def_readonly("gold", &Player::gold)
        .def_readonly("energy", &Player::energy)
        .def_readonly("energy_per_turn", &Player::energyPerTurn)
        .def_readonly("card_draw_per_turn", &Player::cardDrawPerTurn)
        .def_readonly("stance", &Player::stance)
        .def_readonly("orb_slots", &Player::orbSlots)
        .def_readonly("block", &Player::block)
        .def_readonly("artifact", &Player::artifact)
        .def_readonly("dexterity", &Player::dexterity)
        .def_readonly("focus", &Player::focus)
        .def_readonly("strength", &Player::strength)
        .def_readonly("cards_played_this_turn", &Player::cardsPlayedThisTurn)
        .def_readonly("attacks_played_this_turn", &Player::attacksPlayedThisTurn)
        .def_readonly("skills_played_this_turn", &Player::skillsPlayedThisTurn)
        .def_readonly("cards_discarded_this_turn", &Player::cardsDiscardedThisTurn)
        // persistent relic counters (progress toward the next trigger)
        .def_readonly("happy_flower_counter", &Player::happyFlowerCounter)
        .def_readonly("incense_burner_counter", &Player::incenseBurnerCounter)
        .def_readonly("ink_bottle_counter", &Player::inkBottleCounter)
        .def_readonly("inserter_counter", &Player::inserterCounter)
        .def_readonly("nunchaku_counter", &Player::nunchakuCounter)
        .def_readonly("pen_nib_counter", &Player::penNibCounter)
        .def_readonly("sundial_counter", &Player::sundialCounter)
        .def("has_status", &Player::hasStatusRuntime)
        .def("get_status", &Player::getStatusRuntime)
        .def("has_relic", &Player::hasRelicRuntime);

    // ==================== Monster ====================
    pybind::class_<Monster>(m, "Monster")
        .def_readonly("idx", &Monster::idx)
        .def_readonly("id", &Monster::id)
        .def_readonly("cur_hp", &Monster::curHp)
        .def_readonly("max_hp", &Monster::maxHp)
        .def_readonly("block", &Monster::block)
        .def_readonly("strength", &Monster::strength)
        .def_readonly("vulnerable", &Monster::vulnerable)
        .def_readonly("weak", &Monster::weak)
        .def_readonly("poison", &Monster::poison)
        .def_readonly("artifact", &Monster::artifact)
        .def_readonly("half_dead", &Monster::halfDead)
        .def("get_name", &Monster::getName)
        .def("is_alive", &Monster::isAlive)
        .def("is_targetable", &Monster::isTargetable)
        .def("is_dying", &Monster::isDying)
        .def("is_escaping", &Monster::isEscaping)
        .def("is_dead_or_escaped", &Monster::isDeadOrEscaped)
        .def("is_half_dead", &Monster::isHalfDead)
        .def("is_attacking", &Monster::isAttacking)
        .def("get_move_id", [](const Monster &m) {   // current intent's move (0 = INVALID/none)
            return static_cast<int>(m.moveHistory[0]);
        })
        .def("has_status", &Monster::hasStatusInternal)
        .def("get_status", &Monster::getStatusInternal)
        .def("get_move_damage", [](const Monster &m, const BattleContext &bc) {
            return m.getMoveBaseDamage(bc);
        })
        // Per-hit damage the player would actually take from this intent: base damage
        // run through Strength/Weak/Vulnerable/stance/relic modifiers. -1 if not an
        // attack. (get_move_damage returns only the unmodified base.)
        .def("get_move_damage_to_player", [](const Monster &m, const BattleContext &bc) {
            auto di = m.getMoveBaseDamage(bc);
            if (di.damage <= 0) return -1;
            return m.calculateDamageToPlayer(bc, di.damage);
        });

    // ==================== DamageInfo ====================
    pybind::class_<DamageInfo>(m, "DamageInfo")
        .def_readonly("damage", &DamageInfo::damage)
        .def_readonly("attack_count", &DamageInfo::attackCount);

    // ==================== MonsterGroup ====================
    pybind::class_<MonsterGroup>(m, "MonsterGroup")
        .def_readonly("monsters_alive", &MonsterGroup::monstersAlive)
        .def_readonly("monster_count", &MonsterGroup::monsterCount)
        .def("get_alive_count", &MonsterGroup::getAliveCount)
        .def("get_targetable_count", &MonsterGroup::getTargetableCount)
        .def("get_first_targetable", &MonsterGroup::getFirstTargetable)
        .def("are_monsters_basically_dead", &MonsterGroup::areMonstersBasicallyDead)
        .def("__getitem__", [](const MonsterGroup &g, int idx) -> const Monster& {
            if (idx < 0 || idx >= g.monsterCount) {
                throw pybind::index_error("Monster index out of range");
            }
            return g.arr[idx];
        }, pybind::return_value_policy::reference_internal)
        .def("__len__", [](const MonsterGroup &g) { return g.monsterCount; })
        .def("__iter__", [](const MonsterGroup &g) {
            return pybind::make_iterator(g.arr.begin(), g.arr.begin() + g.monsterCount);
        }, pybind::keep_alive<0, 1>());

    // ==================== CardManager ====================
    pybind::class_<CardManager>(m, "CardManager")
        .def_readonly("cards_in_hand", &CardManager::cardsInHand)
        .def_property_readonly("hand", [](const CardManager &cm) {
            return std::vector<CardInstance>(cm.hand.begin(), cm.hand.begin() + cm.cardsInHand);
        })
        .def_property_readonly("draw_pile", [](const CardManager &cm) {
            return std::vector<CardInstance>(cm.drawPile.begin(), cm.drawPile.end());
        })
        .def_property_readonly("discard_pile", [](const CardManager &cm) {
            return std::vector<CardInstance>(cm.discardPile.begin(), cm.discardPile.end());
        })
        .def_property_readonly("exhaust_pile", [](const CardManager &cm) {
            return std::vector<CardInstance>(cm.exhaustPile.begin(), cm.exhaustPile.end());
        })
        .def_property_readonly("draw_pile_size", [](const CardManager &cm) {
            return cm.drawPile.size();
        })
        .def_property_readonly("discard_pile_size", [](const CardManager &cm) {
            return cm.discardPile.size();
        })
        .def_property_readonly("exhaust_pile_size", [](const CardManager &cm) {
            return cm.exhaustPile.size();
        });

    // ==================== CardSelectInfo ====================
    pybind::class_<CardSelectInfo>(m, "CardSelectInfo")
        .def_readonly("pick_count", &CardSelectInfo::pickCount)
        .def_readonly("can_pick_zero", &CardSelectInfo::canPickZero)
        .def_readonly("can_pick_any_number", &CardSelectInfo::canPickAnyNumber)
        .def_readonly("card_select_task", &CardSelectInfo::cardSelectTask)
        .def_property_readonly("cards", [](const CardSelectInfo &csi) {
            return std::vector<CardId>(csi.cards.begin(), csi.cards.end());
        });

    // ==================== search::Action ====================
    pybind::class_<search::Action>(m, "Action")
        .def(pybind::init<search::ActionType>())
        .def(pybind::init<search::ActionType, int>())
        .def(pybind::init<search::ActionType, int, int>())
        .def("__eq__", &search::Action::operator==)
        .def("__ne__", &search::Action::operator!=)
        .def("get_action_type", &search::Action::getActionType)
        .def("get_source_idx", &search::Action::getSourceIdx)
        .def("get_target_idx", &search::Action::getTargetIdx)
        .def("get_select_idx", &search::Action::getSelectIdx)
        .def("is_valid", &search::Action::isValidAction)
        .def("execute", &search::Action::execute)
        .def_static("enumerate_actions", &search::Action::enumerateCardSelectActions)
        .def("__repr__", [](const search::Action &a, const BattleContext &bc) {
            std::ostringstream oss;
            a.printDesc(oss, bc);
            return "<Action " + oss.str() + ">";
        });

    // ==================== BattleContext ====================
    pybind::class_<BattleContext>(m, "BattleContext")
        .def(pybind::init<>())
        .def(pybind::init<const BattleContext&>())
        .def_readonly("seed", &BattleContext::seed)
        .def_readonly("floor_num", &BattleContext::floorNum)
        .def_readonly("encounter", &BattleContext::encounter)
        .def_readonly("ascension", &BattleContext::ascension)
        .def_readonly("outcome", &BattleContext::outcome)
        .def_readonly("input_state", &BattleContext::inputState)
        .def_readonly("turn", &BattleContext::turn)
        .def_readonly("is_battle_over", &BattleContext::isBattleOver)
        .def_readonly("end_turn_queued", &BattleContext::endTurnQueued)
        .def_readonly("turn_has_ended", &BattleContext::turnHasEnded)
        .def_readonly("potion_count", &BattleContext::potionCount)
        .def_readonly("potion_capacity", &BattleContext::potionCapacity)
        .def_readonly("player", &BattleContext::player)
        .def_readonly("monsters", &BattleContext::monsters)
        .def_readonly("cards", &BattleContext::cards)
        .def_readonly("card_select_info", &BattleContext::cardSelectInfo)
        .def_property_readonly("potions", [](const BattleContext &bc) {
            return std::vector<Potion>(bc.potions.begin(), bc.potions.begin() + bc.potionCapacity);
        })
        // Battle actions
        .def("end_turn", &BattleContext::endTurn)
        .def("execute_actions", &BattleContext::executeActions)
        // Re-seed the battle RNGs so a cloned context samples a fresh stochastic
        // outcome (reshuffles, random card effects, monster AI). Does NOT reorder the
        // current draw pile -- its order is part of state. Used by MCTS chance nodes.
        .def("reseed", [](BattleContext &bc, std::uint64_t seed) {
            bc.aiRng = sts::Random(seed);
            bc.cardRandomRng = sts::Random(seed + 1);
            bc.miscRng = sts::Random(seed + 2);
            bc.monsterHpRng = sts::Random(seed + 3);
            bc.potionRng = sts::Random(seed + 4);
            bc.shuffleRng = sts::Random(seed + 5);
        }, pybind::arg("seed"))
        // Re-randomize the (hidden) draw-pile ORDER in place via Fisher-Yates. The
        // draw pile's contents are known to the agent but its order is not, so MCTS
        // determinizes it with the search RNG before sampling a draw -- this is what
        // de-privileges "which cards get drawn". A pure reorder; pile contents/counts
        // are untouched.
        .def("shuffle_draw_pile", [](BattleContext &bc, std::uint64_t seed) {
            auto &dp = bc.cards.drawPile;
            sts::Random r(seed);
            for (int i = static_cast<int>(dp.size()) - 1; i > 0; --i) {
                int j = r.random(i);   // [0, i] inclusive
                std::swap(dp[i], dp[j]);
            }
        }, pybind::arg("seed"))
        .def("drink_potion", &BattleContext::drinkPotion, pybind::arg("idx"), pybind::arg("target") = 0)
        .def("discard_potion", &BattleContext::discardPotion)
        .def("is_card_play_allowed", &BattleContext::isCardPlayAllowed)
        // Total damage a hand card would deal to a target: per-hit base run through
        // Strength/Vigor/Weak/stance/relics + the target's Vulnerable, times the hit count.
        // -1 if not an attack. Per-card bases that scale with state (Body Slam = block,
        // Heavy Blade = +Strength, Perfected Strike = +Strikes, Searing Blow / Rampage /
        // Ritual Dagger, Mind Blast) and multi-hit counts (Twin Strike, Pummel, Sword
        // Boomerang, Whirlwind) mirror playCardImpl exactly; everything else uses the
        // static base table. (Vigor is added inside calculateCardDamage, so bases here
        // must NOT pre-add it -- matching the single-target attack path.) Remaining
        // hand-dependent exceptions: Fiend Fire / Second Wind (per-exhaust).
        .def("get_card_damage", [](const BattleContext &bc, int handIdx, int targetIdx) {
            const auto &c = bc.cards.hand[handIdx];
            const bool up = c.isUpgraded();
            const auto &p = bc.player;
            int base, hits = 1;
            switch (c.getId()) {
                case CardId::BODY_SLAM:        base = p.block; break;
                case CardId::HEAVY_BLADE:      base = 14 + (up ? 4 : 2) * p.strength; break;
                case CardId::PERFECTED_STRIKE: base = 6 + bc.cards.strikeCount * (up ? 3 : 2); break;
                case CardId::SEARING_BLOW: { int n = c.getUpgradeCount(); base = n * (n + 7) / 2 + 12; break; }
                case CardId::RAMPAGE:          base = 8 + c.specialData; break;
                case CardId::RITUAL_DAGGER:    base = c.specialData; break;
                case CardId::MIND_BLAST:       base = static_cast<int>(bc.cards.drawPile.size()); break;
                case CardId::TWIN_STRIKE:      base = up ? 7 : 5; hits = 2; break;
                case CardId::PUMMEL:           base = 2; hits = up ? 5 : 4; break;
                case CardId::SWORD_BOOMERANG:  base = 3; hits = up ? 4 : 3; break;
                case CardId::WHIRLWIND:        base = up ? 8 : 5; hits = p.energy; break;
                default:                       base = getBaseDamage(c.getId(), up); break;
            }
            if (base < 0) return -1;
            return hits * bc.calculateCardDamage(c, targetIdx, base);
        }, pybind::arg("hand_idx"), pybind::arg("target_idx"))
        // Block a given base block resolves to (Dexterity/Frail/No-Block applied). The
        // base block per card has no sim table, so the caller supplies it.
        .def("calculate_card_block", [](const BattleContext &bc, int baseBlock) {
            return bc.calculateCardBlock(baseBlock);
        }, pybind::arg("base_block"))
        // Card select actions
        .def("choose_armaments_card", &BattleContext::chooseArmamentsCard)
        .def("choose_codex_card", &BattleContext::chooseCodexCard)
        .def("choose_discard_to_hand_card", &BattleContext::chooseDiscardToHandCard)
        .def("choose_discovery_card", &BattleContext::chooseDiscoveryCard)
        .def("choose_dual_wield_card", &BattleContext::chooseDualWieldCard)
        .def("choose_exhaust_one_card", &BattleContext::chooseExhaustOneCard)
        .def("choose_exhume_card", &BattleContext::chooseExhumeCard)
        .def("choose_forethought_card", &BattleContext::chooseForethoughtCard)
        .def("choose_headbutt_card", &BattleContext::chooseHeadbuttCard)
        .def("choose_recycle_card", &BattleContext::chooseRecycleCard)
        .def("choose_warcry_card", &BattleContext::chooseWarcryCard)
        // Initialize from GameContext
        .def("init", static_cast<void (BattleContext::*)(const GameContext&)>(&BattleContext::init),
             "Initialize battle context from game context")
        .def("init_with_encounter", static_cast<void (BattleContext::*)(const GameContext&, MonsterEncounter)>(&BattleContext::init),
             "Initialize battle context with specific encounter")
        .def("exit_battle", &BattleContext::exitBattle,
             "Exit battle and update game context")
        .def("__repr__", [](const BattleContext &bc) {
            std::ostringstream oss;
            oss << bc;
            return oss.str();
        });

    // ==================== GameContext ====================
    pybind::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind::init<CharacterClass, std::uint64_t, int>())
        // Existing methods
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    // Basic properties
    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)
        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        .def_readwrite("cur_room", &GameContext::curRoom)
        .def_readwrite("cur_event", &GameContext::curEvent)
        .def_readwrite("boss", &GameContext::boss)
        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)
        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)
        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)
        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard)
        .def_readwrite("potion_count", &GameContext::potionCount)
        .def_readwrite("potion_capacity", &GameContext::potionCapacity);

    // Potions access
    gameContext.def_property_readonly("potions", [](const GameContext &gc) {
        return std::vector<Potion>(gc.potions.begin(), gc.potions.begin() + gc.potionCapacity);
    });

    // Map access
    gameContext.def_property_readonly("map", [](const GameContext &gc) {
        return gc.map.get();
    }, pybind::return_value_policy::reference_internal);

    // ScreenStateInfo access
    gameContext.def_property_readonly("neow_rewards", [](const GameContext &gc) {
        return gc.info.neowRewards;
    });
    gameContext.def_property_readonly("boss_relics", [](const GameContext &gc) {
        return std::vector<RelicId>(std::begin(gc.info.bossRelics), std::end(gc.info.bossRelics));
    });
    gameContext.def_property_readonly("shop", [](const GameContext &gc) -> const Shop& {
        return gc.info.shop;
    }, pybind::return_value_policy::reference_internal);
    gameContext.def_property_readonly("card_select_screen_type", [](const GameContext &gc) {
        return gc.info.selectScreenType;
    });
    gameContext.def_property_readonly("to_select_count", [](const GameContext &gc) {
        return gc.info.toSelectCount;
    });
    gameContext.def_property_readonly("to_select_cards", [](const GameContext &gc) {
        std::vector<Card> cards;
        for (const auto& sc : gc.info.toSelectCards) {
            cards.push_back(sc.card);
        }
        return cards;
    });

    // === CONTROL METHODS ===

    // Map navigation
    gameContext.def("transition_to_map_node", &GameContext::transitionToMapNode,
        "Navigate to a map node. Pass the x coordinate of the node to travel to.");

    // Neow options
    gameContext.def("choose_neow_option", &GameContext::chooseNeowOption,
        "Choose one of the 4 Neow starting bonus options");

    // Boss relic selection
    gameContext.def("choose_boss_relic", &GameContext::chooseBossRelic,
        "Choose a boss relic after defeating a boss (0, 1, or 2)");

    // Event choices
    gameContext.def("choose_event_option", &GameContext::chooseEventOption,
        "Choose an event option by index");

    // Card select screen (transforms, upgrades, etc.)
    gameContext.def("choose_select_card_screen_option", &GameContext::chooseSelectCardScreenOption,
        "Select a card on card select screens (transforms, upgrades, removes, etc.)");

    // Campfire/rest site
    gameContext.def("choose_campfire_option", &GameContext::chooseCampfireOption,
        "Choose a rest site option (0=rest/heal, 1=smith/upgrade, 2=dig, 3=recall, etc.)");

    // Match and Keep
    gameContext.def("choose_match_and_keep_cards", &GameContext::chooseMatchAndKeepCards,
        "Choose two cards for Match and Keep event");

    // Treasure room
    gameContext.def("choose_treasure_room_option", &GameContext::chooseTreasureRoomOption,
        "Choose whether to open treasure chest (true=open, false=leave)");

    // Open treasure chest directly
    gameContext.def("open_treasure_room_chest", &GameContext::openTreasureRoomChest,
        "Open the treasure chest in a treasure room");

    // Enter battle
    gameContext.def("enter_battle", &GameContext::enterBattle,
        "Enter a battle with the specified encounter");

    // Check if has relic
    gameContext.def("has_relic", &GameContext::hasRelic,
        "Check if the player has a specific relic");

    // Potions
    gameContext.def("obtain_potion", &GameContext::obtainPotion,
        "Obtain a potion");
    gameContext.def("drink_potion_at_idx", &GameContext::drinkPotionAtIdx,
        "Drink potion at the specified inventory index");
    gameContext.def("discard_potion_at_idx", &GameContext::discardPotionAtIdx,
        "Discard potion at the specified inventory index");

    // HP management
    gameContext.def("damage_player", &GameContext::damagePlayer,
        "Deal damage to the player");
    gameContext.def("player_lose_hp", &GameContext::playerLoseHp,
        "Player loses HP (different from damage)");
    gameContext.def("player_heal", &GameContext::playerHeal,
        "Heal the player");
    gameContext.def("player_increase_max_hp", &GameContext::playerIncreaseMaxHp,
        "Increase player's max HP");
    gameContext.def("lose_max_hp", &GameContext::loseMaxHp,
        "Reduce player's max HP");

    // Gold management
    gameContext.def("obtain_gold", &GameContext::obtainGold,
        "Obtain gold");
    gameContext.def("lose_gold", &GameContext::loseGold, pybind::arg("amount"), pybind::arg("in_shop") = false,
        "Lose gold");

    // Relics
    gameContext.def("obtain_relic", &GameContext::obtainRelic,
        "Obtain a relic, returns false if already owned");

    // Keys
    gameContext.def("obtain_key", &GameContext::obtainKey,
        "Obtain a key (for Act 4)");
    gameContext.def("has_key", &GameContext::hasKey,
        "Check if player has a specific key");

    // Get available map paths
    gameContext.def("get_available_map_paths", [](const GameContext &gc) {
        std::vector<int> paths;
        if (gc.curMapNodeY == -1) {
            // At the start, get all starting nodes
            for (int x = 0; x < 7; ++x) {
                if (gc.map->getNode(x, 0).edgeCount > 0) {
                    paths.push_back(x);
                }
            }
        } else {
            // Get available paths from current node
            const auto& node = gc.map->getNode(gc.curMapNodeX, gc.curMapNodeY);
            for (int i = 0; i < node.edgeCount; ++i) {
                paths.push_back(node.edges[i]);
            }
        }
        return paths;
    }, "Get list of valid x coordinates for map navigation");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
