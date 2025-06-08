import type { Botella } from '../types/Botella';

export default function BotellaListaValida({ botellas }: { botellas: Botella[] }) {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4 text-green-600">✅ Botellas válidas</h2>
      <ul className="space-y-2">
        {botellas.map(b => (
          <li key={b.id} className="p-3 rounded border border-green-300 bg-green-50">
            <strong>{b.nombre}</strong> - {b.capacidad}ml - {b.color}
          </li>
        ))}
      </ul>
    </div>
  );
}
