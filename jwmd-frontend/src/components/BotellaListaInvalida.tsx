import type { Botella } from '../types/Botella';

export default function BotellaListaInvalida({ botellas }: { botellas: Botella[] }) {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4 text-red-600">❌ Botellas inválidas</h2>
      <ul className="space-y-2">
        {botellas.map(b => (
          <li key={b.id} className="p-3 rounded border border-red-300 bg-red-50">
            <strong>{b.nombre}</strong> - {b.capacidad}ml - {b.color}
            <div className="text-sm text-red-700 italic">Razón: {b.razonInvalidez}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
