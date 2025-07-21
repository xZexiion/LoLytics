import { get_game_data } from "./game.js";
import fs from "fs";
import { API_KEYS } from "./api_keys.js";
import { get_ids } from "./get_match_ids.js";

async function get_batch(match_ids) {
	const promises = [];
	for (let i = 0; i < match_ids.length; i++) {
		promises.push(get_game_data(match_ids[i], API_KEYS[i]));
	}
	const results = await Promise.all(promises);
	const batch = [];
	for (const result of results) {
		batch.push(result);
	}
	return batch.filter((e) => e != null);
}

async function download_games(rank) {
	let match_ids = await get_ids(rank);
	match_ids = new Set(match_ids);
	match_ids = Array.from(match_ids);

	console.log(`Processing ${match_ids.length} matches`);

	let idx = 0;
	for (let i = 0; i < match_ids.length; i += API_KEYS.length) {
		let batch = await get_batch(match_ids.slice(i, i + API_KEYS.length));
		for (const game of batch) {
			idx++;
			fs.mkdirSync(`match_data/${rank}/game_${idx}`);
			for (let j = 0; j < game.length; j++) {
				fs.writeFileSync(
					`match_data/${rank}/game_${idx}/${j}.json`,
					JSON.stringify(game[j]),
				);
			}
		}
		console.log(`Processed ${idx} matches`);
	}
}

async function main() {
	const ranks = [
		"IRON",
		"BRONZE",
		"SILVER",
		"GOLD",
		"PLATINUM",
		"EMERALD",
		"DIAMOND",
		"MASTER",
		"GRANDMASTER",
		"CHALLENGER",
	];

	fs.mkdirSync(`match_data`);
	for (const rank of ranks) {
		fs.mkdirSync(`match_data/${rank}`);
		await download_games(rank);
	}
}

main();
