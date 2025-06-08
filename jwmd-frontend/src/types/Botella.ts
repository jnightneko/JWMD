export interface Botella {
  id: string;
  nombre: string;
  capacidad: number;
  color: string;
  clasificacion: 'valida' | 'invalida';
  razonInvalidez?: string;
}

