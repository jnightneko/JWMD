import axios from 'axios';
import type { Botella } from '../types/Botella';

const API_URL = 'http://localhost:3000/jwmd'; // Ajusta el puerto si tu backend usa otro

export const getBotellas = () => axios.get<Botella[]>(API_URL);
