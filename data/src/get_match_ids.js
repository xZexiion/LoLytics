import { api_call } from "./utils.js";
import { api_keys } from "./api_keys.js";

function shuffle(array) {
	let current_index = array.length, random_index;

	while (current_index > 0) {
		random_index = Math.floor(Math.random() * current_index);
		current_index--;
		[array[current_index], array[random_index]] = [
			array[random_index], array[current_index]];
	}

	return array;
}

async function get_summoner_ids(rank, tier, page, key) {
	const summoner_ids = [];
	if (rank == 'CHALLENGER') {
		let response = await api_call(`https://euw1.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key=${key}`);
		response = await response.json();
		for (const summoner of response.entries) {
			summoner_ids.push(summoner.summonerId);
		}
	} else if (rank == 'GRANDMASTER') {
		let response = await api_call(`https://euw1.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=${key}`);
		response = await response.json();
		for (const summoner of response.entries) {
			summoner_ids.push(summoner.summonerId);
		}
	} else if (rank == 'MASTER') {
		let response = await api_call(`https://euw1.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5?api_key=${key}`);
		response = await response.json();
		for (const summoner of response.entries) {
			summoner_ids.push(summoner.summonerId);
		}
	} else {
		let response = await api_call(`https://euw1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/${rank}/${tier}?page=${page}&api_key=${key}`);
		response = await response.json();
		for (const summoner of response) {
			summoner_ids.push(summoner.summonerId);
		}
	}
	shuffle(summoner_ids);
	return summoner_ids;
}

async function get_summoner_match_ids(summoner_id, key) {
	let puuid = await api_call(`https://euw1.api.riotgames.com/lol/summoner/v4/summoners/${summoner_id}?api_key=${key}`);
	puuid = await puuid.json();
	puuid = puuid.puuid;

	const now = new Date();
	const one_month_ago = new Date(now);
	one_month_ago.setMonth(now.getMonth() - 1);
	const start = Math.floor(one_month_ago.getTime() / 1000);

	let match_history = await api_call(`https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/${puuid}/ids?startTime=${start}&queue=420&start=0&count=50&api_key=${key}`)
	match_history = await match_history.json();

	return match_history;
}

async function get_match_id_batch(rank, tier, page, key) {
	let match_ids = [];
	const limit = 250;

	const summoner_ids = await get_summoner_ids(rank, tier, page, key);
	for (const summoner_id of summoner_ids) {
		match_ids = match_ids.concat(await get_summoner_match_ids(summoner_id, key));
		if (match_ids.length >= limit) {
			break;
		}
	}

	return match_ids;
}

export async function get_ids(rank) {
	const promises = [];

	const tiers = ['I', 'II', 'III', 'IV'];
	let idx = 0;
	for (const tier of tiers) {
		for (let i = 0; i < 10; i++) {
			promises.push(get_match_id_batch(rank, tier, i + 1, api_keys[idx]));
			idx++;
		}
	}

	const results = await Promise.all(promises);
	const match_ids = [].concat(...results);

	shuffle(match_ids);
	return match_ids;
}
