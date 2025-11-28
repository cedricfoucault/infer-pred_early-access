import {parseURLParams} from './urlParams.js';

const urlParams = parseURLParams();

export const VERSION_CHANGE_POINT = "cp";
export const VERSION_RANDOM_WALK = "rw";

export let version = urlParams.version ? urlParams.version : VERSION_CHANGE_POINT;
